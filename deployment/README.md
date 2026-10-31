# Triển khai CA-TCC + DINOv3: Docker, CI/CD và web miễn phí

## 1. Kiến trúc và phạm vi

Dự án cung cấp hai đường sử dụng cùng model đã huấn luyện:

- **Docker/API**: Python và PyTorch CPU; nhận WAV qua HTTP; phù hợp tích hợp vào ứng dụng khác.
- **Web tĩnh**: ONNX Runtime Web chạy model ngay trên thiết bị người dùng. Hosting chỉ phân phối HTML, JavaScript, WASM và trọng số; không cần máy cá nhân luôn bật.

SpeechCommands có 35 lớp, nhận tối đa 1 giây. RAVDESS có 8 lớp cảm xúc, nhận tối đa 6 giây. Dữ liệu đi qua tiền xử lý và chuẩn hóa tương ứng với lúc huấn luyện. Không huấn luyện lại khi phát hành code.

Luồng CI: push vào main / pull request -> test -> kết quả kiểm tra.

Luồng release: tag v* hoặc chạy thủ công -> xác minh model -> test -> build -> kiểm tra API container với WAV thật -> push Docker Hub -> pull và kiểm tra lại -> deploy GitHub Pages nếu đã bật.

Web cũ được giữ nguyên tại https://ca-tcc-audio-studio-vu-20261018.royalcream8.chatgpt.site . GitHub Pages là bản demo bổ sung do người dùng đã đồng ý tạo. Chỉ gọi CD hoàn tất sau khi workflow deploy thành công.

## 2. Chuẩn bị tài khoản và môi trường

1. Dùng tài khoản có quyền ghi repository GitHub của dự án.
2. Tạo tài khoản Docker Hub miễn phí, xác minh email, tạo repository **public** tên `ca-tcc-audio`.
3. Trong Docker Hub, tạo access token cho phát hành với quyền đọc/ghi phù hợp; nhập trực tiếp vào GitHub Secret `DOCKERHUB_TOKEN`. Không đưa token vào code, chat, ảnh chụp hoặc PDF.
4. Cài Docker Engine hoặc Docker Desktop. Nếu Docker đã cài thì giữ cấu hình hiện tại. Script `deployment/install-docker-ubuntu.sh` chỉ dành cho Ubuntu và cần nhập mật khẩu quản trị trực tiếp trong terminal.
5. Kiểm tra bằng `docker version`, `docker compose version` và `docker info`.
6. GitHub CLI là công cụ tùy chọn cho thao tác tương đương giao diện; đăng nhập bằng `gh auth login`.

## 3. Chuẩn bị model release

Model lớn không được commit trực tiếp vào Git. Từ môi trường đã có PyTorch và toàn bộ artifacts, chạy:

```bash
.venv-dinov3/bin/python scripts/deployment_assets.py prepare
```

Đầu ra:

- `build/models/`: TorchScript, metadata và normalization của hai model.
- `build/browser/`: manifest và chunks ONNX cho web.
- `build/fixtures/`: hai WAV validation không sửa đổi và kết quả tham chiếu để kiểm tra suy luận.
- `build/deployment-manifest.json`: SHA256 của các file phát hành.
- `output/deployment/deployment-assets.zip` và file SHA256 đi kèm.

Script kiểm tra danh tính backbone, số lớp, checksum model và chunks. Archive có đường dẫn cố định và timestamp ZIP cố định. Các WAV chỉ dùng kiểm tra tương đương số học, không dùng suy ra accuracy/F1.

Upload ZIP và SHA256 vào một GitHub Release có version riêng, ví dụ `model-assets-v1`. Nếu mạng không ổn định với tệp lớn, chạy `python scripts/deployment_assets.py split`, upload các tệp `deployment-assets.part-*`, `deployment-assets.index.json` và SHA256; đặt MODEL_ASSETS_URL trỏ tới index JSON. Mỗi phần 8 MiB có checksum riêng; runner ghép lại và vẫn kiểm tra SHA256 toàn bộ ZIP. Khi đổi version release, truyền `--release-base-url` tương ứng cho lệnh split. Giữ nguyên tài nguyên của release đã phát hành. Khi thay model, tạo release mới và cập nhật SHA256; không thay nội dung cũ mà giữ nguyên version.

Đặt repository variables trong Settings -> Secrets and variables -> Actions:

| Tên | Giá trị |
|---|---|
| DOCKERHUB_USERNAME | Username Docker Hub thực tế |
| MODEL_ASSETS_URL | URL HTTPS của ZIP hoặc index JSON chia phần từ release công khai |
| MODEL_ASSETS_SHA256 | 64 ký tự SHA256 từ file đi kèm |
| ENABLE_PAGES | true sau khi bật Pages và muốn triển khai web |

Chỉ `DOCKERHUB_TOKEN` là secret. Các biến trên không chứa mật khẩu.

Runner sạch tải và kiểm tra như sau:

```bash
python scripts/deployment_assets.py fetch --url "$MODEL_ASSETS_URL" --sha256 "$MODEL_ASSETS_SHA256"
```

Không tin file tải về chỉ vì tên đúng. Script kiểm tra hash của archive trước giải nén, từ chối đường dẫn thoát thư mục, symlink, mục trùng, file ngoài manifest và nội dung không khớp hash.

## 4. Build và chạy Docker trên máy

Sau khi có `build/models/`:

```bash
docker build --progress=plain -t ca-tcc-audio:local .
docker run -d --name ca-tcc-demo -p 127.0.0.1:8000:8000 ca-tcc-audio:local
```

Mở http://127.0.0.1:8000 . Kiểm tra `docker logs ca-tcc-demo` và `docker inspect ca-tcc-demo` khi cần.

Dockerfile dùng Python 3.11, PyTorch CPU 2.5.1, dependency inference cố định và base image được ghim digest. Model được nạp trước khi mở cổng. Container chạy bằng UID 10001, lưu job trong `/jobs`; Docker volume giữ dữ liệu runtime. Không cần GPU, source DINOv3 hoặc pretrained checkpoint gốc cho inference TorchScript.

`.dockerignore` chỉ đưa đúng source inference, web, dependency, giấy phép và model bundles vào build context. Không đưa dataset huấn luyện, môi trường ảo hoặc credential.

Ví dụ gọi API:

```bash
curl --fail http://127.0.0.1:8000/healthz
curl --fail http://127.0.0.1:8000/api/models
curl --fail -X POST 'http://127.0.0.1:8000/api/predict?dataset=RAVDESS' \
  -H 'Content-Type: audio/wav' --data-binary @build/fixtures/RAVDESS.wav
```

Body POST là WAV nguyên bản, không phải multipart. Kết quả gồm dataset, backbone, hash phiên bản, thời lượng và top_k. Điểm softmax không phải xác suất đã hiệu chỉnh.

## 5. Kiểm tra container thật

```bash
python3 scripts/smoke_audio_api.py --fixtures build/fixtures --output output/deployment/container-verification.json
```

Bài kiểm tra phải xác nhận:

- Health check thành công và hai model sẵn sàng.
- Hai model trả điểm hữu hạn trong [0,1].
- WAV thật cho nhãn đứng đầu giống tham chiếu, sai khác điểm top-k dưới 0.0001.
- Dataset không hợp lệ và file lỗi bị từ chối.
- SpeechCommands từ chối WAV dài hơn 1 giây.
- Origin ngoài danh sách bị từ chối.

Không dùng kết quả này để báo cáo chất lượng tổng quát của model. Accuracy/F1 cần tập đánh giá có nhãn đủ lớn và quy trình đánh giá riêng.

Chạy kiểm tra logic:

```bash
python -m unittest discover -s tests -p test_batch.py -v
python -m unittest discover -s tests -p test_server_deployment.py -v
python -m unittest discover -s tests -p test_deployment_assets.py -v
```

## 6. CI trên GitHub

Workflow `.github/workflows/ci.yml` chạy khi push main, mở/cập nhật pull request hoặc được kích hoạt thủ công. Workflow cài dependency nhẹ, kiểm tra ZIP, báo cáo batch, hợp đồng HTTP và tính an toàn của model artifacts. Không dùng token Docker Hub trong pull request.

Mở tab Actions -> Audio CI -> lần chạy cụ thể. Ghi lại URL, commit, thời gian và kết quả thực tế. Dấu kiểm xanh chỉ được ghi nhận khi toàn bộ job cần thiết thành công.

## 7. Phát hành Docker Hub

Workflow `.github/workflows/release.yml` chạy bằng tag `v*` hoặc nút Run workflow. Chỉ phát hành code đã được chọn và kiểm tra.

```bash
git tag v1.0.0
git push origin v1.0.0
```

Workflow tải model theo checksum, chạy test, build image, chạy container và test WAV thật. Sau đó đăng nhập Docker Hub bằng secret, push chính image đã test với tag `sha-<full-commit>` và tag phiên bản nếu được kích hoạt bằng tag Git.

Workflow pull image đã công bố, lưu RepoDigest, chạy API lần nữa và lưu bằng chứng thành artifact. Đây là kiểm tra sau phát hành; máy nhận có thể tái sử dụng các layer đã tải. Kiểm tra trên một máy/daemon sạch là bước xác nhận độc lập bổ sung khi có môi trường.

Mọi URL Docker Hub và digest trong báo cáo phải lấy từ kết quả thực tế. Không ghi placeholder như thể image đã được công bố.

## 8. CD cho GitHub Pages

1. Repository công khai trên gói GitHub Free.
2. Mở Settings -> Pages -> Build and deployment -> Source -> GitHub Actions.
3. Đặt `ENABLE_PAGES=true`.
4. Chạy workflow phát hành đã cấu hình đầy đủ.

Script `scripts/build_browser_release.py` kết hợp source web với chunks đã xác minh, tạo `build/web-dist/`, `.nojekyll` và `deployment.json`. File deployment ghi commit và hash model để truy vết.

Job Pages chỉ chạy sau khi job phát hành thành công. `actions/upload-pages-artifact` lưu bộ web hoàn chỉnh; `actions/deploy-pages` xuất URL thực tế. Không viết URL dự kiến thành kết quả đã xác nhận.

Để chạy thử bản web độc lập trước khi publish:

```bash
python scripts/build_browser_release.py --revision local-review
python -m http.server 8011 --bind 127.0.0.1 --directory build/web-dist
```

Nếu thư mục đầu ra đã tồn tại, chọn thư mục mới bằng `--output`. Script chủ động không ghi đè một release đang được kiểm tra.

Mở web, chọn WAV thật, xác nhận rồi chạy. Kiểm tra cả hai tác vụ, giao diện điện thoại, tải CSV, WAV hỏng và âm thanh quá dài. Trình duyệt tự xử lý âm thanh; lần tải model đầu khoảng 84 MiB mỗi tác vụ. Thiết bị yếu có thể chậm hoặc thiếu RAM.

Để chứng minh không phụ thuộc máy cá nhân: mở URL Pages từ thiết bị khác sau khi dừng container/server local, rồi tắt máy cá nhân và dự đoán bằng một phiên trình duyệt mới. Ghi nhận thao tác kiểm tra thực tế; không coi việc mở được trang chủ là đã kiểm tra inference.

## 9. Quay lại bản trước

Docker: dùng tag phiên bản hoặc digest đã xác minh trước đó; đổi `AUDIO_IMAGE`, pull và chạy lại Compose.

```bash
export AUDIO_IMAGE='USERNAME/ca-tcc-audio:v1.0.0'
docker compose pull
docker compose up -d
```

`USERNAME` là giá trị người dùng phải thay bằng namespace đã phát hành, không phải tên image có sẵn.

Web: chọn lại commit và model asset/checksum của bản tốt trước đó, chạy workflow phát hành thủ công trên ref tương ứng, rồi kiểm tra `deployment.json` và inference. Không sửa hoặc xóa tag/version đã công bố để che lịch sử.

Nếu deploy thất bại trước khi công bố bản mới, giữ nguyên bản web đang chạy. Nếu bản mới đã được công bố nhưng có lỗi, thực hiện quy trình phát hành lại bản tốt và ghi lại nguyên nhân.

## 10. Giới hạn và xử lý lỗi

- Docker daemon không chạy: mở Docker Desktop hoặc kiểm tra service; không thay quyền socket thành 666.
- Quyền truy cập socket: dùng đúng Docker context hoặc sudo theo kiểu cài đặt; không tắt bảo vệ hệ thống.
- Thiếu model: chạy prepare trên máy có artifacts hoặc fetch theo URL/checksum đã cấu hình.
- Hash sai: dừng phát hành, kiểm tra bản release; không bỏ kiểm tra SHA256.
- HTTP 403: kiểm tra Host và Origin. Khi đổi cổng ngoài hoặc tên miền, cấu hình `--allowed-host` và `--allowed-origin` đúng địa chỉ; không mở wildcard tùy tiện.
- HTTP 400: kiểm tra WAV, dataset và thời lượng.
- Docker Hub unauthorized: kiểm tra username, repository và token; không in secret ra log.
- Pages chưa bật: kiểm tra Settings -> Pages và quyền pages:write/id-token:write.
- CI không thấy model: model không nằm trong Git source; cần URL release assets công khai có thể tải không cần phiên đăng nhập cá nhân.
- Web thiếu chunks hoặc WASM: kiểm tra toàn bộ build artifact và đường dẫn tương đối dưới subpath repository.
- Giới hạn miễn phí có thể thay đổi; xem quota tài khoản trước khi tăng tần suất build hoặc lưu artifact.

## 11. Tài liệu chính thức

- Docker + GitHub Actions: https://docs.docker.com/build/ci/github-actions/
- Docker Engine Ubuntu: https://docs.docker.com/engine/install/ubuntu/
- GitHub Pages custom workflow: https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages
- GitHub Pages limits: https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits
- DINOv3 license: https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md
- RAVDESS: https://zenodo.org/records/1188976
- Speech Commands: https://arxiv.org/abs/1804.03209

## 12. Sinh lại PDF từ bằng chứng đã lưu

```bash
python -m pip install -r deployment/requirements-report.txt
python scripts/report_deployment.py
```

Đầu ra là `output/pdf/BAO_CAO_CICD_DOCKER_AI.pdf`. Trình tạo đọc kết quả API, ONNX, container và `publication.json` trong `output/deployment/`. Không tự đánh dấu thành công các bước chưa có bằng chứng. PDF phải được render và kiểm tra trực quan trước khi nộp.

Nếu cần upload chia phần trên mạng không ổn định, tạo release nháp rồi chạy `python scripts/upload_model_release.py`. Trình upload chỉ xử lý release nháp, bỏ qua phần đã tải thành công và giữ trạng thái để có thể tiếp tục. Khi đủ các phần, kiểm tra danh sách asset, rồi mới công bố release và cấu hình MODEL_ASSETS_URL trỏ tới index JSON.
