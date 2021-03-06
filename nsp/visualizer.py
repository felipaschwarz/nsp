import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

class Visualizer:
    """Plot the generated data."""

    def _get_min_max(matrix):
        vmin = min(matrix.view(-1))
        vmax = max(matrix.view(-1))
        return vmin, vmax

    def _get_global_min_max(matrix_list):
        vmin = 0
        vmax = 0
        for m in matrix_list:
            new_vmin, new_vmax = Visualizer._get_min_max(m)
            vmin = min(vmin, new_vmin)
            vmax = max(vmax, new_vmax)
        return vmin, vmax

    def _get_norm(matrix_list):
        vmin, vmax = Visualizer._get_global_min_max(matrix_list)
        absmax = max(abs(vmin), abs(vmax))
        if absmax == 0.:
            absmax = 1.e-307
        vmin = -absmax
        vmax = absmax
        norm = matplotlib.colors.TwoSlopeNorm(vmin=vmin, vcenter=0., vmax=vmax)
        return norm

    def visualize_pattern(activations, pdf_filepath, scale='layerscale', cmap_style='viridis'):
        """
        Visualize an activation pattern.

        Layers are visualized on seperate pages.
        Channels are visualized as seperately on one page.

        Parameters
        ----------
        activations : Activations
            Activations to visualize.
        pdf_filepath : str
            Pdf path to store the visualization.
        scale : {'layerscale', 'layernorm', 'globalscale', 'globalnorm'}, default 'layerscale'
            Scaling of the colomap.
                - 'layerscale'  : provides one color scale per activation layer.
                - 'layernorm'   : provides one color scale per activation layer
                                    and sets the center of the scale to 0.
                - 'globalscale' : provides one color scale per activation pattern.
                - 'globalnorm'  : provides one color scale per activation pattern
                                    and sets the center of the scale to 0.
                                    
            The options 'layernorm' and 'globalnorm' are particularly useful for diverging colormap styles.
        cmap_style : str, default 'viridis'
            Pick a `cmap_style` from the matplotlib colormaps
            (https://matplotlib.org/stable/tutorials/colors/colormaps.html).

        Returns
        -------
        None
        """
        os.makedirs(os.path.dirname('./'+pdf_filepath), exist_ok=True)
        with PdfPages(pdf_filepath) as pdf:
            globalnorm = Visualizer._get_norm(activations.layeractivations)
            globalvmin, globalvmax = Visualizer._get_global_min_max(activations.layeractivations)

            for index_layer, (layeractivation, layername) in enumerate(zip(activations.layeractivations, activations.layernames)):
                layervmin, layervmax = Visualizer._get_global_min_max(layeractivation)
                layernorm = Visualizer._get_norm(layeractivation)

                max_cols = 4
                if (len(layeractivation.shape) == 1):
                    layeractivation  = layeractivation.unsqueeze(0).unsqueeze(0)
                n_channels = layeractivation.shape[0]
                n_cols = min(n_channels, max_cols)

                fig, axes = plt.subplots(-(-n_channels // max_cols), n_cols, sharey=True, subplot_kw={'xticks': []})
                fig.suptitle(f'Layer {index_layer}: {layername}')
                #fig.suptitle(f'Layer {index_layer+1}')
                for i, ax in enumerate(np.array(axes).reshape(-1)):
                    if i < n_channels:
                        if scale == 'standard':
                            cmap = ax.imshow(layeractivation[i], aspect='equal', cmap=plt.get_cmap(cmap_style))
                        elif scale == 'layernorm':
                            cmap = ax.imshow(layeractivation[i], aspect='equal', cmap=plt.get_cmap(cmap_style), norm=layernorm)
                        elif scale == 'layerscale':
                            cmap = ax.imshow(layeractivation[i], aspect='equal', cmap=plt.get_cmap(cmap_style), vmin=layervmin, vmax=layervmax)
                        elif scale == 'globalnorm':
                            cmap = ax.imshow(layeractivation[i], aspect='equal', cmap=plt.get_cmap(cmap_style), norm=globalnorm)
                        elif scale == 'globalscale':
                            cmap = ax.imshow(layeractivation[i], aspect='equal', cmap=plt.get_cmap(cmap_style), vmin=globalvmin, vmax=globalvmax)
                        elif scale == 'globalPositive':
                            cmap = ax.imshow(layeractivation[i], aspect='equal', cmap=plt.get_cmap(cmap_style), vmin=0, vmax=globalvmax)
                        else:
                            raise NotImplementedError
                    else:
                        ax.axis('off')
                cax = fig.add_axes([0.2,0.05,0.6,0.02])
                cbar = fig.colorbar(cmap, cax=cax, orientation='horizontal')
                cbar.ax.tick_params(labelsize=10)

                pdf.savefig(fig, bbox_inches='tight', pad_inches=0.5)
                plt.close()
