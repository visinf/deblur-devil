# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import sys

from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter


# NOTE: summary operations are registered at module level on instantiation of a SummaryWriter
#   summary.scalar
#   summary.image
#   ...


class SummaryWriter:
    def __init__(self, logdir, flush_secs=120):

        self.writer = TensorboardSummaryWriter(
            log_dir=logdir,
            purge_step=None,
            max_queue=10,
            flush_secs=flush_secs,
            filename_suffix='')

        self.global_step = None
        self.active = True

        # ------------------------------------------------------------------------
        # register add_* and set_* functions in summary module on instantiation
        # ------------------------------------------------------------------------
        this_module = sys.modules[__name__]
        list_of_names = dir(SummaryWriter)
        for name in list_of_names:

            # add functions (without the 'add' prefix)
            if name.startswith('add_'):
                setattr(this_module, name[4:], getattr(self, name))

            #  set functions
            if name.startswith('set_'):
                setattr(this_module, name, getattr(self, name))

    def set_global_step(self, value):
        self.global_step = value

    def set_active(self, value):
        self.active = value

    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None):
        if self.active:
            global_step = self.global_step if global_step is None else global_step
            self.writer.add_audio(
                tag, snd_tensor, global_step=global_step, sample_rate=sample_rate, walltime=walltime)

    def add_custom_scalars(self, layout):
        if self.active:
            self.writer.add_custom_scalars(layout)

    def add_custom_scalars_marginchart(self, tags, category='default', title='untitled'):
        if self.active:
            self.writer.add_custom_scalars_marginchart(tags, category=category, title=title)

    def add_custom_scalars_multilinechart(self, tags, category='default', title='untitled'):
        if self.active:
            self.writer.add_custom_scalars_multilinechart(tags, category=category, title=title)

    def add_embedding(self, mat, metadata=None, label_img=None, global_step=None,
                      tag='default', metadata_header=None):
        if self.active:
            global_step = self.global_step if global_step is None else global_step
            self.writer.add_embedding(
                mat, metadata=metadata, label_img=label_img, global_step=global_step,
                tag=tag, metadata_header=metadata_header)

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        if self.active:
            global_step = self.global_step if global_step is None else global_step
            self.writer.add_figure(
                tag, figure, global_step=global_step, close=close, walltime=walltime)

    def add_graph(self, model, input_to_model=None, verbose=False):
        if self.active:
            self.writer.add_graph(model, input_to_model=input_to_model, verbose=verbose)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None):
        if self.active:
            global_step = self.global_step if global_step is None else global_step
            self.writer.add_histogram(
                tag, values, global_step=global_step, bins=bins,
                walltime=walltime, max_bins=max_bins)

    def add_histogram_raw(self, tag, min, max, num, sum, sum_squares,
                          bucket_limits, bucket_counts, global_step=None,
                          walltime=None):
        if self.active:
            global_step = self.global_step if global_step is None else global_step
            self.writer.add_histogram_raw(
                tag, min=min, max=max, num=num, sum=sum, sum_squares=sum_squares,
                bucket_limits=bucket_limits, bucket_counts=bucket_counts,
                global_step=global_step, walltime=walltime)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        if self.active:
            global_step = self.global_step if global_step is None else global_step
            self.writer.add_image(
                tag, img_tensor, global_step=global_step, walltime=walltime, dataformats=dataformats)

    def add_image_with_boxes(self, tag, img_tensor, box_tensor, global_step=None,
                             walltime=None, rescale=1, dataformats='CHW'):
        if self.active:
            global_step = self.global_step if global_step is None else global_step
            self.writer.add_image_with_boxes(
                tag, img_tensor, box_tensor,
                global_step=global_step, walltime=walltime,
                rescale=rescale, dataformats=dataformats)

    def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'):
        if self.active:
            global_step = self.global_step if global_step is None else global_step
            self.writer.add_images(
                tag, img_tensor, global_step=global_step, walltime=walltime, dataformats=dataformats)

    def add_mesh(self, tag, vertices, colors=None, faces=None, config_dict=None, global_step=None, walltime=None):
        if self.active:
            global_step = self.global_step if global_step is None else global_step
            self.writer.add_mesh(
                tag, vertices, colors=colors, faces=faces, config_dict=config_dict,
                global_step=global_step, walltime=walltime)

    def add_onnx_graph(self, graph):
        if self.active:
            self.writer.add_onnx_graph(graph)

    def add_pr_curve(self, tag, labels, predictions, global_step=None,
                     num_thresholds=127, weights=None, walltime=None):
        if self.active:
            global_step = self.global_step if global_step is None else global_step
            self.writer.add_pr_curve(
                tag, labels, predictions, global_step=global_step,
                num_thresholds=num_thresholds, weights=weights, walltime=walltime)

    def add_pr_curve_raw(self, tag, true_positive_counts,
                         false_positive_counts,
                         true_negative_counts,
                         false_negative_counts,
                         precision,
                         recall,
                         global_step=None,
                         num_thresholds=127,
                         weights=None,
                         walltime=None):
        if self.active:
            global_step = self.global_step if global_step is None else global_step
            self.writer.add_pr_curve_raw(
                tag, true_positive_counts,
                false_positive_counts,
                true_negative_counts,
                false_negative_counts,
                precision,
                recall,
                global_step=global_step,
                num_thresholds=num_thresholds,
                weights=weights,
                walltime=walltime)

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if self.active:
            global_step = self.global_step if global_step is None else global_step
            self.writer.add_scalar(
                tag, scalar_value, global_step=global_step, walltime=walltime)

    def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None):
        if self.active:
            global_step = self.global_step if global_step is None else global_step
            self.writer.add_scalars(
                main_tag, tag_scalar_dict, global_step=global_step, walltime=walltime)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        if self.active:
            global_step = self.global_step if global_step is None else global_step
            self.writer.add_text(
                tag, text_string, global_step=global_step, walltime=walltime)

    def add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None):
        if self.active:
            global_step = self.global_step if global_step is None else global_step
            self.writer.add_video(
                tag, vid_tensor, global_step=global_step, fps=fps, walltime=walltime)

    def close(self):
        self.writer.close()

    def __enter__(self):
        return self.writer.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.writer.__exit__(exc_type, exc_val, exc_tb)
