"""Draw the original stage objectives in a plain black-and-white layout."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def main():
    out = Path(__file__).resolve().parents[1] / 'docs/figures/dinov3_original'
    out.mkdir(parents=True, exist_ok=True)
    rows = [
        ('Self-supervised', r'$L = (L_{T1} + L_{T2}) + 0.7\,L_{\mathrm{NT\!-\!Xent}}$',
         'Backbone + projection + TC'),
        ('SupCon', r'$L = 0.01\,(L_{T1} + L_{T2}) + 0.1\,L_{\mathrm{SupCon}}$',
         'Backbone + projection + TC'),
        ('Classification', r'$L = \mathrm{CrossEntropy}(\mathrm{logits}, y)$',
         'Linear evaluation: classifier\nFine-tuning: encoder + classifier'),
    ]
    with plt.rc_context({'font.family': 'Liberation Serif', 'text.color': 'black',
                         'mathtext.fontset': 'stix', 'svg.fonttype': 'none',
                         'savefig.facecolor': 'white'}):
        fig, ax = plt.subplots(figsize=(15, 3.8))
        ax.set_xlim(0, 15)
        ax.set_ylim(0, 3.8)
        ax.axis('off')
        cols = [(0.08, 2.65), (3.14, 6.22), (9.77, 5.12)]
        for (x, w), title in zip(cols, ['Training stage', 'Loss', 'Updated parameters']):
            ax.text(x+w/2, 3.48, title, ha='center', va='center',
                    fontsize=14, fontweight='bold', color='black')
        for i, (stage, formula, updates) in enumerate(rows):
            y = 2.43 - i * 1.02
            for x, w in cols:
                ax.add_patch(Rectangle((x, y), w, .78, facecolor='white',
                                       edgecolor='black', linewidth=.9))
            for (x, w), value, size in zip(cols, [stage, formula, updates], [13, 15, 12.5]):
                ax.text(x+w/2, y+.39, value, ha='center', va='center',
                        fontsize=size, linespacing=1.6, color='black')
            for (x, w), (next_x, _) in zip(cols, cols[1:]):
                ax.annotate('', xy=(next_x-.03, y+.39), xytext=(x+w+.03, y+.39),
                            arrowprops={'arrowstyle': '-|>', 'lw': .9,
                                        'color': 'black', 'shrinkA': 0, 'shrinkB': 0})
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        for ext in ('png', 'svg'):
            fig.savefig(out / f'loss_paths.{ext}', dpi=200,
                        bbox_inches='tight', pad_inches=.05)
        plt.close(fig)


if __name__ == '__main__':
    main()
