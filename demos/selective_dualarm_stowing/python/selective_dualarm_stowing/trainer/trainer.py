import tempfile

import chainer
from chainer.training import extensions


def get_trainer(
        gpu,
        model,
        optimizer,
        iter_train,
        iter_val,
        max_iter,
        out=None,
        resume=None,
        interval_log=10,
        interval_eval=1000,
        interval_save=10000,
        log_header=None
):

    if out is None:
        out = tempfile.mktemp()

    updater = chainer.training.StandardUpdater(
        iter_train, optimizer, device=gpu)
    trainer = chainer.training.Trainer(
        updater, (max_iter, 'iteration'), out=out)

    trainer.extend(
        extensions.Evaluator(iter_val, model, device=gpu),
        trigger=(interval_eval, 'iteration'),
        invoke_before_training=True,
    )

    model_name = model.__class__.__name__
    trainer.extend(
        extensions.dump_graph('main/loss', out_name='%s.dot' % model_name))
    trainer.extend(
        extensions.snapshot(
            savefun=chainer.serializers.hdf5.save_hdf5,
            filename='%s_trainer_iter_{.updater.iteration}.h5' % model_name),
        trigger=(interval_save, 'iteration'))
    trainer.extend(
        extensions.snapshot_object(
            model,
            savefun=chainer.serializers.hdf5.save_hdf5,
            filename='%s_model_iter_{.updater.iteration}.h5' % model_name),
        trigger=(interval_save, 'iteration'))
    trainer.extend(extensions.LogReport(
        trigger=(interval_log, 'iteration'), log_name='log.json'))
    trainer.extend(extensions.PrintReport(log_header))
    trainer.extend(extensions.ProgressBar(update_interval=1))

    if resume:
        if resume.endswith('npz'):
            chainer.serializers.load_npz(resume, trainer)
        else:
            chainer.serializers.load_hdf5(resume, trainer)

    return trainer
