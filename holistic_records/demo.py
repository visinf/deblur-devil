# Author: Jochen Gast <jochen.gast@visinf.tu-darmstadt.de>

import matlab_tools
import numpy as np
import recorder
import torch


def main():
    im = matlab_tools.imread("./test.jpg").astype(np.float32) * 1.0 / 255.0
    im = matlab_tools.rgb2gray(im)

    h, w = im.shape
    c = 1
    b = 8

    images = torch.zeros(b, c, h, w)
    example_index = torch.zeros(b, 1).long()
    example_basename = ["test"] * b

    for j in range(b):
        images[j, ...] = torch.from_numpy(im.transpose([0, 1]))
        example_index[j] = j
        example_basename[j] = "image-%03i" % j

    class arguments():
        pass

    args = arguments()
    args.save = "./logdir"
    epoch_recorder = recorder.EpochRecorder(args,
                                            epoch=4,
                                            dataset="MNISTTrain",
                                            png=True)
    epoch_recorder.add_image(
        example_basename,
        images,
        cmap="jet")


#     loss_dict = { "xe": float(epoch), "top1": float(epoch), "top5": float(epoch)}
#     epoch_recorder.add_scalars(name="training_losses", value=loss_dict)


# epoch_recorder = EpochRecorder(root="./logdir",
#                                epoch=1,
#                                dataset="MNISTTrain",
#                                write_png=True,
#                                write_csv=True,
#                                write_mat=True)

# for name in ["output1", "output2"]:
#     for example in range(3):
#         img = im + np.random.randn(im.shape[0], im.shape[1])
#         epoch_recorder.add_image(name=name, image=img, imagesc=True, cmap="gray", example=example)


# epoch_recorder = EpochRecorder(root="./logdir",
#                                epoch=2,
#                                dataset="MNISTTrain",
#                                write_png=True,
#                                write_csv=True,
#                                write_mat=True)

# for name in ["output1", "output2"]:
#     for example in range(3):
#         img = im + np.random.randn(im.shape[0], im.shape[1])
#         epoch_recorder.add_image(name=name, image=img, imagesc=True, cmap="gray", example=example)

if __name__ == '__main__':
    main()
