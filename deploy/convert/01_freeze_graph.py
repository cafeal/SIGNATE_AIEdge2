import argparse
import pathlib

import albumentations as albu

import tensorflow as tf
import tensorflow.keras as K

import cv2

from TF_CenterNet.datasets.common_transforms import get_common_transforms


def freeze_graph(args):
    # clear session
    K.backend.clear_session()
    K.backend.set_learning_phase(0)
    # load model
    model = K.models.load_model(args.model_file, compile=False)
    # freeze graph
    sess = K.backend.get_session()
    graph_def = sess.graph.as_graph_def()
    pruned_graph_def = tf.graph_util.remove_training_nodes(graph_def)
    constant_graph = tf.graph_util.convert_variables_to_constants(
        sess,
        pruned_graph_def,
        [output.op.name for output in model.outputs],
        [v.op.name for v in model.variables],
    )
    # save
    tf.io.write_graph(constant_graph, args.output_dir.as_posix(), args.graph_name, as_text=False)

    # validation(args, model)


def validation(args, model):
    transform = albu.Compose(get_common_transforms())
    img = cv2.imread(args.image_path.as_posix())
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(image=img)
    inputs = [img]

    sess = K.backend.get_session()
    orig_ret = sess.run(model.output, {model.input: inputs})

    with tf.Session() as sess:
        graph = sess.graph
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(args.output_dir / args.graph_name, "rb") as f:
            graph_def.ParseFromString(f.read())

        ph_in = tf.placeholder(tf.float32, shape=(None, 320, 512, 3), name="input")
        with graph.as_default():
            op_outputs = tf.import_graph_def(
                graph_def, {"input_6": ph_in}, ["detector_model/output/BiasAdd"],
            )

        t_output = op_outputs[0].outputs[0]
        frozen_ret = sess.run(t_output, {ph_in: inputs})

    mean_diff = abs(orig_ret - frozen_ret).mean()
    print("diff :", mean_diff)
    assert mean_diff < 1e-4


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-file",
        "-m",
        help="keras hdf5 model",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--graph-name",
        "-g",
        help="output graph name",
        default="frozen_graph.pb",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="result_01",
        help="output directory",
        type=pathlib.Path,
    )
    parser.add_argument("--data-dir", help="data directory", default="../data")
    args = parser.parse_args()

    freeze_graph(args)
