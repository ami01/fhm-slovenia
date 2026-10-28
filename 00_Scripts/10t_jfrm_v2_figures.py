"""10t — JFRM v2 figures (review 2026-10-09). Reuses 10s; changes only Fig. 9(c): agreement of the four learners'
binary maps (09k) instead of five products on the Very High quintile. Writes to Manuscript/JFRM_v2/figures/main/.

    python 00_Scripts/10t_jfrm_v2_figures.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
_spec = importlib.util.spec_from_file_location("s10", os.path.join(os.path.dirname(os.path.abspath(__file__)), "10s_jem_final_figures.py"))
s10 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(s10)
globals().update({k: v for k, v in vars(s10).items() if not k.startswith("__")})
V2_M = os.path.join(ROOT, 'Manuscript', 'JFRM_v2', 'figures', 'main')


def v2_fig_maps():
    """Probability (a) and binary (b) maps: four learners 2x2, ensemble enlarged."""
    os.makedirs(V2_M, exist_ok=True)
    learners = ['RF', 'XGBoost', 'CatBoost', 'DNN']
    probs = {m: _map(os.path.join(PROB, SIZE, 'FSM_%s_probability.tif' % m)) for m in MODELS}
    bins = {m: _map(os.path.join(CLSF, SIZE, 'FSM_%s_binary.tif' % m)) for m in MODELS}
    h, w = probs['RF'].shape
    asp = w / h
    W, gap, th, lab, key = FIG_W, 0.05, 0.17, 0.20, 0.42
    sw = (W - th * asp - 4 * gap) / 4            # small map width
    sh = sw / asp
    ew = 2 * sw + th * asp                       # ensemble width: spans two small rows
    eh = ew / asp
    blk = lab + 2 * (sh + th) + key
    hw = 0.66 * W                                # agreement map, row (c)
    hh = hw / asp
    row3 = lab + th + hh + 0.54
    H = 2 * blk + row3 + 0.05
    fig = plt.figure(figsize=(W, H))
    cols = ['#E8EAF0', '#1A5FB4']

    def ax_at(x, y, ww, hh):
        return fig.add_axes([x / W, y / H, ww / W, hh / H])

    for b, (arrs, letter) in enumerate(((probs, 'a'), (bins, 'b'))):
        top = H - b * blk
        fig.text(0.004, (top - 0.03) / H, '(%s)' % letter, fontsize=8.5, fontweight='bold', va='top')
        y_row = [top - lab - th - sh, top - lab - 2 * (th + sh)]
        for i, m in enumerate(learners):
            ax = ax_at(gap / 2 + (i % 2) * (sw + gap), y_row[i // 2], sw, sh)
            if b == 0:
                im = ax.imshow(arrs[m], cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
            else:
                ax.imshow(arrs[m], cmap=ListedColormap(cols), norm=BoundaryNorm([-.5, .5, 1.5], 2),
                          interpolation='nearest')
            ax.set_title(LABEL[m], fontsize=6.4, fontweight='bold', pad=1.5); ax.axis('off')
        ex = W - ew - gap / 2
        ey = y_row[1]
        ax = ax_at(ex, ey, ew, eh)
        if b == 0:
            ax.imshow(arrs['Ensemble_mean'], cmap='RdYlBu_r', vmin=0, vmax=1, interpolation='nearest')
        else:
            ax.imshow(arrs['Ensemble_mean'], cmap=ListedColormap(cols),
                      norm=BoundaryNorm([-.5, .5, 1.5], 2), interpolation='nearest')
        ax.set_title('Ensemble mean', fontsize=7.4, fontweight='bold', pad=2); ax.axis('off')
        ky = ey - key + 0.05
        if b == 0:
            cax = ax_at(W / 2 - 1.1, ky + 0.16, 2.2, 0.07)
            cb = fig.colorbar(im, cax=cax, orientation='horizontal')
            cb.set_label('susceptibility probability', fontsize=6.3, labelpad=1)
            cb.ax.tick_params(labelsize=5.8, length=2, pad=1)
        else:
            lax = ax_at(0, ky, W, 0.3); lax.axis('off')
            lax.legend(handles=[mpatches.Patch(facecolor=c, edgecolor='0.4', lw=0.4, label=l)
                                for c, l in zip(cols, ['not flood-prone', 'flood-prone'])],
                       loc='center', ncol=2, fontsize=6.4, frameon=False)
    # row (c): agreement of the FOUR learners' binary maps (09k); the ensemble is their mean, not a fifth vote
    top = H - 2 * blk
    y3 = top - lab - th - hh
    fig.text(0.004, (top - 0.03) / H, '(c)', fontsize=8.5, fontweight='bold', va='top')
    # distinct hue and lightness per count (viridis steps); 0 stays a neutral grey
    cl = ['#E6E6E6', '#FDE725', '#5EC962', '#3B528B', '#440154']
    # national share of each agreement count: the table behind the 76.1 % / 16.4 % in the text
    st = pd.read_csv(os.path.join(A, 'overflag_v2', 'agreement4.csv')).sort_values('n_models')
    pct = 100 * st.share.values
    names = ['0 of 4', '1', '2', '3', 'all 4']
    x = gap
    ax = ax_at(x, y3, hw, hh)
    ax.imshow(_map(os.path.join(A, 'overflag_v2', 'agreement4_binary.tif')), cmap=ListedColormap(cl),
              norm=BoundaryNorm(np.arange(-0.5, 5), 5), interpolation='nearest')
    ax.set_title('Learners classifying the cell flood-prone', fontsize=7.4, fontweight='bold', pad=2)
    ax.axis('off')
    # share of national area per count, beside the map
    bx0 = x + hw + 0.42
    # the 0-of-5 class is the grey background; it is described in the caption, not keyed
    bax = ax_at(bx0, y3 + 0.18 * hh, W - bx0 - 0.40, 0.64 * hh)
    yy = np.arange(4)[::-1]
    bax.barh(yy, pct[1:], color=cl[1:], edgecolor='0.35', lw=0.4, height=0.72)
    for yv, v in zip(yy, pct[1:]):
        bax.text(v + 0.4, yv, '%.1f %%' % v, va='center', ha='left', fontsize=5.9)
    bax.set_yticks(yy); bax.set_yticklabels(names[1:], fontsize=6.0)
    bax.set_xlim(0, 20); bax.set_xticks([0, 10, 20])
    bax.tick_params(axis='x', labelsize=5.8, length=2, pad=1); bax.tick_params(axis='y', length=0)
    bax.set_xlabel('share of Slovenia (%)', fontsize=6.2, labelpad=1)
    for sp in ('top', 'right'):
        bax.spines[sp].set_visible(False)
    lax = ax_at(0, y3 - 0.50, W, 0.42); lax.axis('off')
    lax.legend(handles=[mpatches.Patch(facecolor=c, edgecolor='0.4', lw=0.4,
                                       label='%s (%.1f %%)' % (n, v))
                        for c, n, v in zip(cl[1:], names[1:], pct[1:])],
               loc='center', ncol=4, fontsize=6.3, frameon=False, handlelength=1.1,
               columnspacing=1.4, handletextpad=0.4)
    save(fig, os.path.join(V2_M, 'Fig09_national_maps.png'))



if __name__ == '__main__':
    v2_fig_maps()
