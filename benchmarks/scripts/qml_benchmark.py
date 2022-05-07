import uuid
import os
import utils


if __name__ == "__main__":
    _uuid = str(uuid.uuid4())
    nwires, nlayer, nitrs, timeLimit, isgpu, minus, path, nbatch = utils.arg(qml=True)
    if isgpu == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    else:
        import tensorflow as tf

        gpu = tf.config.list_physical_devices("GPU")
        tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)

    train_img, test_img, train_lbl, test_lbl = utils.mnist_data_preprocessing(
        nwires - 1
    )

    from qml_pennylane import pennylane_benchmark
    from qml_tc_tf import tensorcircuit_tf_benchmark
    from qml_tc_jax import tensorcircuit_jax_benchmark
    from qml_tfquantum import tfquantum_benchmark

    pl_json = pennylane_benchmark(
        _uuid,
        nwires,
        nlayer,
        nitrs,
        timeLimit,
        isgpu,
        train_img,
        test_img,
        train_lbl,
        test_lbl,
        nbatch,
        False,
    )
    tfq_json = tfquantum_benchmark(
        _uuid,
        nwires,
        nlayer,
        nitrs,
        timeLimit,
        isgpu,
        train_img,
        test_img,
        train_lbl,
        test_lbl,
        nbatch,
    )
    tc32_json = tensorcircuit_tf_benchmark(
        _uuid,
        nwires,
        nlayer,
        nitrs,
        timeLimit,
        isgpu,
        train_img,
        test_img,
        train_lbl,
        test_lbl,
        nbatch,
        "32",
    )
    tc64_json = tensorcircuit_tf_benchmark(
        _uuid,
        nwires,
        nlayer,
        nitrs,
        timeLimit,
        isgpu,
        train_img,
        test_img,
        train_lbl,
        test_lbl,
        nbatch,
        "64",
    )
    tcjax_json = tensorcircuit_jax_benchmark(
        _uuid,
        nwires,
        nlayer,
        nitrs,
        timeLimit,
        isgpu,
        train_img,
        test_img,
        train_lbl,
        test_lbl,
        nbatch,
        "64",
    )
    utils.save([pl_json, tfq_json, tc32_json, tc64_json, tcjax_json], _uuid, path)
