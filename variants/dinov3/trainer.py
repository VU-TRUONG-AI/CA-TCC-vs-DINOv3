"""Isolated baseline trainer copy; only the linear-probe freeze hook is changed.

Source: trainer/trainer.py at variant creation. Losses, optimizers, checkpoint and
validation selection behavior are retained. This module has not been executed.
"""
import os
import json
import random
import time
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss, SupConLoss


def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode, resume=False, evaluate_test=True):
    # Start training
    logger.debug("Training started ....")

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    checkpoint_dir = os.path.join(experiment_log_dir, "saved_models")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float('inf')
    history = []
    supervised_stage = training_mode not in ("self_supervised", "SupCon")
    start_epoch = 1
    resume_path = os.path.join(checkpoint_dir, 'resume.pt')
    if resume and os.path.isfile(resume_path):
        saved = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(saved['model_state_dict'])
        temporal_contr_model.load_state_dict(saved['temporal_contr_model_state_dict'])
        model_optimizer.load_state_dict(saved['model_optimizer'])
        temp_cont_optimizer.load_state_dict(saved['temporal_optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if torch.cuda.is_available() and saved['cuda_rng']:
            torch.cuda.set_rng_state_all([v.cpu() for v in saved['cuda_rng']])
        ns = saved['numpy_rng']
        np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
        random.setstate(saved['python_rng'])
        history, best_loss = saved['history'], saved['best_loss']
        start_epoch = saved['epoch'] + 1
        logger.info(f'Resuming from epoch {start_epoch}')
    for epoch in range(start_epoch, config.num_epoch + 1):
        epoch_start = time.monotonic()
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode)
        valid_loss, valid_acc, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        if (training_mode != "self_supervised") and (training_mode != "SupCon"):
            scheduler.step(valid_loss)

        history.append(dict(epoch=epoch, train_loss=float(train_loss), train_accuracy=float(train_acc),
                            validation_loss=float(valid_loss) if supervised_stage else None,
                            validation_accuracy=float(valid_acc) if supervised_stage else None,
                            epoch_seconds=time.monotonic()-epoch_start))
        with open(os.path.join(experiment_log_dir, 'history.json'), 'w') as stream:
            json.dump(history, stream, indent=2)
        if supervised_stage and valid_loss < best_loss:
            best_loss = valid_loss
            torch.save({'model_state_dict': model.state_dict(),
                        'temporal_contr_model_state_dict': temporal_contr_model.state_dict(),
                        'epoch': epoch, 'validation_loss': float(valid_loss)},
                       os.path.join(checkpoint_dir, 'ckp_best.pt'))

        numpy_state = np.random.get_state()
        state = dict(model_state_dict=model.state_dict(),
                     temporal_contr_model_state_dict=temporal_contr_model.state_dict(),
                     model_optimizer=model_optimizer.state_dict(), temporal_optimizer=temp_cont_optimizer.state_dict(),
                     scheduler=scheduler.state_dict(), epoch=epoch, history=history, best_loss=best_loss,
                     torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     numpy_rng=(numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]), python_rng=random.getstate())
        torch.save(state, resume_path + '.tmp')
        os.replace(resume_path + '.tmp', resume_path)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:2.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:2.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')


    # save the model after training ...
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(),
                'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if supervised_stage:
        selected = torch.load(os.path.join(checkpoint_dir, 'ckp_best.pt'), map_location=device, weights_only=True)
        model.load_state_dict(selected['model_state_dict'])
        with open(os.path.join(experiment_log_dir, 'selection.json'), 'w') as stream:
            json.dump(dict(criterion='minimum validation cross entropy', epoch=selected['epoch'],
                           validation_loss=selected['validation_loss'], checkpoint='saved_models/ckp_best.pt'), stream, indent=2)
        # evaluate on the test set
        if evaluate_test:
            logger.debug('\nEvaluate on the Test set:')
            test_loss, test_acc, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
            logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config,
                device, training_mode):
    total_loss = 0.0
    total_acc = 0
    sample_count = 0
    model.train()
    if 'train_linear' in training_mode or 'tl' in training_mode or training_mode == 'random_init':
        # DINOv3 freezes its complete encoder through configure_stage/train.
        model.encoder.eval()
    temporal_contr_model.train()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised" or training_mode == "SupCon":
            predictions1, features1 = model(aug1)
            predictions2, features2 = model(aug2)

            # normalize projection feature vectors
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            temp_cont_loss1, temp_cont_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_feat2 = temporal_contr_model(features2, features1)


        if training_mode == "self_supervised":
            lambda1 = 1
            lambda2 = 0.7
            nt_xent_criterion = NTXentLoss(device, data.shape[0], config.Context_Cont.temperature,
                                           config.Context_Cont.use_cosine_similarity)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + \
                   nt_xent_criterion(temp_cont_feat1, temp_cont_feat2) * lambda2


        elif training_mode == "SupCon":
            lambda1 = 0.01
            lambda2 = 0.1
            Sup_contrastive_criterion = SupConLoss(device)

            supCon_features = torch.cat([temp_cont_feat1.unsqueeze(1), temp_cont_feat2.unsqueeze(1)], dim=1)
            loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + Sup_contrastive_criterion(supCon_features,
                                                                                             labels) * lambda2

        else:
            output = model(data)
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc += labels.eq(predictions.detach().argmax(dim=1)).sum().item()

        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss in {training_mode}, batch {batch_idx}')
        total_loss += loss.item() * len(labels)
        sample_count += len(labels)
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss /= sample_count

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_acc = 0
    else:
        total_acc /= sample_count
    return total_loss, total_acc


def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])
    sample_count = 0

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            if (training_mode == "self_supervised") or (training_mode == "SupCon"):
                pass
            else:
                output = model(data)

            # compute loss
            if (training_mode != "self_supervised") and (training_mode != "SupCon"):
                predictions, features = output
                loss = criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).sum().item())
                total_loss.append(loss.item() * len(labels))
                sample_count += len(labels)

                pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_loss = 0
        total_acc = 0
        return total_loss, total_acc, [], []
    else:
        total_loss = sum(total_loss) / sample_count
        total_acc = sum(total_acc) / sample_count
        return total_loss, total_acc, outs, trgs




def gen_pseudo_labels(model, dataloader, device, experiment_log_dir):
    from sklearn.metrics import accuracy_score
    model.eval()
    softmax = nn.Softmax(dim=1)

    # saving output data
    all_pseudo_labels = np.array([])
    all_labels = np.array([])
    all_data = []

    with torch.no_grad():
        for data, labels, _, _ in dataloader:
            data = data.float().to(device)
            labels = labels.view((-1)).long().to(device)

            # forward pass
            predictions, features = model(data)

            normalized_preds = softmax(predictions)
            pseudo_labels = normalized_preds.max(1, keepdim=True)[1].squeeze()
            all_pseudo_labels = np.append(all_pseudo_labels, pseudo_labels.cpu().numpy())

            all_labels = np.append(all_labels, labels.cpu().numpy())
            all_data.append(data.cpu())

    all_data = torch.cat(all_data, dim=0)

    data_save = dict()
    data_save["samples"] = all_data
    data_save["labels"] = torch.from_numpy(all_pseudo_labels).long()
    file_name = f"pseudo_train_data.pt"
    torch.save(data_save, os.path.join(experiment_log_dir, file_name))
    with open(os.path.join(experiment_log_dir, 'pseudo_label_diagnostics.json'), 'w') as f:
        json.dump({'count': len(all_pseudo_labels),
                   'diagnostic_accuracy': accuracy_score(all_labels, all_pseudo_labels),
                   'predicted_class_counts': np.bincount(all_pseudo_labels.astype(int)).tolist()}, f, indent=2)

    print("Pseudo labels generated ...")
