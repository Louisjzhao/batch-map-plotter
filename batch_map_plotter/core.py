import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import numpy as np
from shapely.geometry import Point
import mapclassify
import contextily as cx
import warnings

# Default color for missing values (semi-transparent gray)
DEFAULT_COLOR = (0.85, 0.85, 0.85, 0.4)

# Ensure minus signs render correctly
mpl.rcParams['axes.unicode_minus'] = False

def get_non_overlapping_labels(df, min_dist=3000):
    """
    Remove overlapping labels based on centroid distance (in meters).
    """
    shown = []
    coords = []
    for _, row in df.iterrows():
        pt = row['centroid']
        if not coords:
            shown.append(True)
            coords.append(pt)
        else:
            too_close = any(pt.distance(p) < min_dist for p in coords)
            shown.append(not too_close)
            if not too_close:
                coords.append(pt)
    return shown

def generate_color_map(levels, order_by='good_to_bad', palette=None, alpha=0.7):
    """
    Generate a color map for given levels.
    """
    n = len(levels)
    if palette is None:
        base = plt.cm.get_cmap('RdYlBu', n)
        colors = [base(i) for i in range(n)]
    elif isinstance(palette, str):
        base = plt.cm.get_cmap(palette, n)
        colors = [base(i) for i in range(n)]
    else:
        colors = [mpl.colors.to_rgba(c, alpha=alpha) for c in palette]

    if order_by == 'bad_to_good':
        colors = colors[::-1]

    return dict(zip([str(lv) for lv in levels], colors)), colors

def bin_numeric_variable(series, bins=5, strategy='natural_breaks'):
    """
    Bin a numeric series using the specified strategy.
    """
    if strategy == 'quantiles':
        binned, bins = pd.qcut(series, q=bins, retbins=True, duplicates='drop')
    elif strategy == 'equal_interval':
        binned, bins = pd.cut(series, bins=bins, retbins=True, duplicates='drop')
    elif strategy == 'natural_breaks':
        scheme = mapclassify.NaturalBreaks(y=series.dropna(), k=bins)
        binned = pd.cut(series, bins=scheme.bins, include_lowest=True)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    return binned.astype(str), bins

def extract_left_bound(val):
    """
    Extract the left bound of an interval for sorting.
    """
    try:
        if isinstance(val, pd.Interval):
            return val.left
        else:
            return float(str(val).split(',')[0].replace('(', '').replace('[', ''))
    except Exception:
        return np.nan

def plot_batch_maps(
    gdf,                        #GeoDataFrame
    vars,                       #List for variables to plot

    # === Variable Configuration ===
    var_config=None,            # Dict with individual variable config (type, bins, palette, etc.)
    group_by=None,              # Column to split maps into groups (e.g. categories)
    bin_by_group=True,          # Whether to bin numeric variables separately within each group
    data_type='mixed',         # Default data type: 'numeric', 'categorical', or 'mixed'

    # === Geometry Settings ===
    geometry_type='polygon',   # 'polygon' or 'point'
    point_size =30,
    basemap=None,              # contextily tile provider (e.g., contextily.providers.Gaode.Normal)
    layer_alpha=1.0,           # Transparency for map layer (0 to 1)

    # === Binning & Colors ===
    bins=5,                    # Number of bins for numeric variables
    binning_strategy='natural_breaks',  # Binning strategy for numeric vars
    palette='RdYlBu',          # Default colormap or list of colors
    reverse_colormap=False,    # Reverse the order of color mapping
    reverse_legend=False,      # Reverse legend display order
    alpha=0.7,                 # Alpha transparency for color mapping

    # === Labeling Options ===
    show_labels=True,          # Whether to draw text labels
    label_col='name',          # Column to use as label text
    label_min_dist=3000,       # Minimum distance (in meters) to avoid label overlap
    label_fontsize=8,          # Label font size
    fontfamily='Arial',  # Font family for all text (label, title, legend)

    # === Output Settings ===
    output_dir='.',            # Folder to save figures
    dpi=300,                   # Resolution of output image
    figsize=(10, 10),          # Figure size in inches
    return_updated_gdf=False   # Whether to return modified GeoDataFrame with bin columns
):
    """
    Batch plot variables on a map, with optional grouping, binning, labeling and basemap support.
    """
    gdf = gdf.copy()
    if gdf.crs.to_epsg() != 3857:
        gdf = gdf.to_crs(epsg=3857)

    gdf['centroid'] = gdf.geometry.centroid

    if group_by:
        groups = gdf[group_by].dropna().unique()
    else:
        groups = [None]

    for group in groups:
        sub_gdf = gdf if group is None else gdf[gdf[group_by] == group].copy()

        for var in vars:
            print(f"🛠 Processing variable: {var}  (group: {group})")

            cfg = var_config.get(var, {}) if var_config else {}
            var_type = cfg.get('type', data_type)

            if var_type == 'mixed' or var_type is None:
                if pd.api.types.is_numeric_dtype(sub_gdf[var]):
                    var_type = 'numeric'
                else:
                    var_type = 'categorical'


            print(f"Determined variable type: {var_type}")

            if var_type == 'categorical':
                levels = cfg.get('order') or cfg.get('levels') or sorted(pd.unique(sub_gdf[var].dropna()))
                order_by = cfg.get('order_by', 'good_to_bad')  # ✅ separate color direction
                custom_palette = cfg.get('palette', palette)


                cmap, color_seq = generate_color_map(levels[::-1] if reverse_colormap else levels, order_by, custom_palette, alpha)

                nan_mask = sub_gdf[var].isna() | sub_gdf[var].astype(str).str.lower().eq('nan')
                sub_gdf['fill_color'] = sub_gdf[var].astype(str).map(cmap)
                sub_gdf.loc[nan_mask, 'fill_color'] = pd.Series([DEFAULT_COLOR] * nan_mask.sum(), index=sub_gdf.loc[nan_mask].index)
                sub_gdf['fill_color'] = sub_gdf['fill_color'].apply(lambda x: tuple(x))

            elif var_type == 'numeric':
                bin_col = f"{var}_bin"
                if group is not None:
                    bin_col += f"_{group}"

                n_bins = cfg.get('bins', bins)
                strategy = cfg.get('strategy', binning_strategy)
                custom_palette = cfg.get('palette', palette)

                
                if bin_by_group and group is not None:
                    # Bin separately within each group
                    sub_gdf[bin_col], bin_edges = bin_numeric_variable(sub_gdf[var], bins=n_bins, strategy=strategy)
                else:
                    # Global binning: apply consistent bins across all groups
                    global_series = gdf[var]
                    gdf[bin_col], bin_edges = bin_numeric_variable(global_series, bins=n_bins, strategy=strategy)
                    sub_gdf[bin_col] = gdf.loc[sub_gdf.index, bin_col]

                sorted_bins = sorted(sub_gdf[bin_col].dropna().unique(), key=extract_left_bound)
                dtype = pd.api.types.CategoricalDtype(categories=sorted_bins, ordered=True)
                sub_gdf[bin_col] = sub_gdf[bin_col].astype(dtype)

                if return_updated_gdf:
                    gdf.loc[sub_gdf.index, bin_col] = sub_gdf[bin_col]

                levels = [str(x) for x in dtype.categories]
                print(f"Numeric bins: {levels}")
                print(f"Bin edges: {bin_edges}")

                cmap, color_seq = generate_color_map(levels[::-1] if reverse_colormap else levels, 'good_to_bad', custom_palette, alpha)
                nan_mask = sub_gdf[bin_col].isna() | sub_gdf[bin_col].astype(str).str.lower().eq('nan')

                sub_gdf['fill_color'] = sub_gdf[bin_col].astype(str).map(cmap)
                sub_gdf.loc[nan_mask, 'fill_color'] = pd.Series([DEFAULT_COLOR] * nan_mask.sum(), index=sub_gdf.loc[nan_mask].index)
                sub_gdf['fill_color'] = sub_gdf['fill_color'].apply(lambda x: tuple(x))
            else:
                warnings.warn(f"Unsupported variable type for {var}")
                continue

            fig, ax = plt.subplots(figsize=figsize)

            if geometry_type == 'point':
                sub_gdf.plot(ax=ax, color=sub_gdf['fill_color'], markersize=point_size, edgecolor='black', alpha=layer_alpha)
            else:
                sub_gdf.plot(ax=ax, color=sub_gdf['fill_color'], edgecolor='black', alpha=layer_alpha)

            if show_labels and label_col in sub_gdf.columns:
                sub_gdf['show_label'] = get_non_overlapping_labels(sub_gdf, min_dist=label_min_dist)
                for _, row in sub_gdf.iterrows():
                    if row['show_label']:
                        ax.text(row['centroid'].x, row['centroid'].y, str(row[label_col]),
                                ha='center', va='center', fontsize=label_fontsize,
                                family=fontfamily, color='black')
                        
            if basemap:
                cx.add_basemap(ax, source=basemap)
                ax.set_xlim(sub_gdf.total_bounds[[0, 2]])
                ax.set_ylim(sub_gdf.total_bounds[[1, 3]])


            levels_for_legend = [lv for lv in levels if str(lv).lower() != 'nan' and str(lv).lower() != 'none']
            if reverse_legend:
                levels_for_legend = levels_for_legend[::-1]

            handles = [mpatches.Patch(color=cmap[str(lv)], label=str(lv)) for lv in levels_for_legend if str(lv) in cmap]

            if sub_gdf[var].isna().any() or nan_mask.any():
                handles.append(mpatches.Patch(color=DEFAULT_COLOR, label='No Data'))

            ax.legend(handles=handles, title=var, loc='lower left', fontsize=9, title_fontproperties=mpl.font_manager.FontProperties(family=fontfamily))

            title = f"{var}" + (f" ({group})" if group else "")
            ax.set_title(title, fontsize=16, fontfamily=fontfamily)
            ax.axis('off')

            filename = f"{output_dir}/{group}_{var}_map.jpg" if group else f"{output_dir}/{var}_map.jpg"
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"✅ Saved: {filename}")

    if return_updated_gdf:
        return gdf