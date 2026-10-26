"""Render the seven executed CA-TCC stages as a plain black-and-white flowchart."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    out = Path(__file__).resolve().parents[1] / 'docs/figures/dinov3_original'
    out.mkdir(parents=True, exist_ok=True)
    stages = [
        ('Self-supervised', 'Full training set\nNo labels'),
        ('Linear evaluation', '10% labeled data\nEncoder frozen'),
        ('Fine-tuning', '10% labeled data\nEncoder trainable'),
        ('Pseudo-labeling', 'Full training set\nInference'),
        ('SupCon', 'Full training set\nPseudo-labels'),
        ('Linear evaluation', '10% labeled data\nEncoder frozen'),
        ('Fine-tuning', '10% labeled data\nEncoder trainable'),
    ]
    with plt.rc_context({'font.family': 'Liberation Serif', 'text.color': 'black',
                         'svg.fonttype': 'none', 'savefig.facecolor': 'white'}):
        fig, ax = plt.subplots(figsize=(17.5, 2.1))
        ax.set_xlim(0, 17.5)
        ax.set_ylim(0, 2.1)
        ax.axis('off')
        for i, (title, detail) in enumerate(stages):
            x = .06 + i * 2.51
            ax.add_patch(Rectangle((x, .45), 2.23, 1.2,
                                   facecolor='white', edgecolor='black', linewidth=.9))
            ax.text(x + 1.115, 1.35, title, ha='center', va='center',
                    fontsize=13, fontweight='bold', color='black')
            ax.text(x + 1.115, .89, detail, ha='center', va='center',
                    fontsize=11.5, linespacing=1.5, color='black')
            if i < len(stages) - 1:
                ax.annotate('', xy=(x + 2.49, 1.05), xytext=(x + 2.25, 1.05),
                            arrowprops={'arrowstyle': '-|>', 'lw': .9,
                                        'color': 'black', 'shrinkA': 0, 'shrinkB': 0})
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        for ext in ('png', 'svg'):
            fig.savefig(out / f'training_pipeline.{ext}', dpi=200,
                        bbox_inches='tight', pad_inches=.05)
        plt.close(fig)


if __name__ == '__main__':
    main()
