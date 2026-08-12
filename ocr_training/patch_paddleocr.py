from __future__ import annotations

import argparse
from pathlib import Path

PATCH_MARKER = "KCODE_RECOVERY_CHECKPOINT_PATCH_V1"


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(
            f"PaddleOCR v3.7.0 patch anchor count was {source.count(old)}, expected 1"
        )
    return source.replace(old, new, 1)


def patch_program(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if PATCH_MARKER in source:
        return False
    source = _replace_once(source, "import copy\n", "import copy\nimport signal\n")
    source = _replace_once(
        source,
        "    reader_start = time.time()\n    eta_meter = AverageMeter()\n",
        """    reader_start = time.time()
    eta_meter = AverageMeter()

    # KCODE_RECOVERY_CHECKPOINT_PATCH_V1
    save_batch_step = int(config[\"Global\"].get(\"save_batch_step\", 500))
    recovery_keep = max(2, int(config[\"Global\"].get(\"recovery_keep\", 3)))
    stop_state = {\"requested\": False, \"signal\": None}

    def request_graceful_stop(signum, _frame):
        stop_state[\"requested\"] = True
        stop_state[\"signal\"] = signum
        logger.warning(
            \"received signal %s; saving a recovery checkpoint after this batch\", signum
        )

    if dist.get_rank() == 0:
        signal.signal(signal.SIGTERM, request_graceful_stop)
        signal.signal(signal.SIGINT, request_graceful_stop)

    resume_batch = int(best_model_dict.pop(\"resume_batch\", 0))

    def begin_checkpoint(prefix):
        marker = os.path.join(save_model_dir, prefix + \".complete\")
        if os.path.exists(marker):
            os.remove(marker)

    def mark_checkpoint_complete(prefix):
        marker = os.path.join(save_model_dir, prefix + \".complete\")
        marker_tmp = marker + \".tmp\"
        with open(marker_tmp, \"w\") as marker_file:
            marker_file.write(str(global_step))
            marker_file.flush()
            os.fsync(marker_file.fileno())
        os.replace(marker_tmp, marker)

    def save_recovery_checkpoint(reason):
        if dist.get_rank() != 0:
            return
        prefix = \"recovery_step_{}\".format(global_step)
        begin_checkpoint(prefix)
        resume_state = dict(best_model_dict)
        resume_state.update(
            start_epoch=epoch,
            resume_batch=idx + 1,
            global_step=global_step,
        )
        save_model(
            model,
            optimizer,
            save_model_dir,
            logger,
            config,
            is_best=False,
            prefix=prefix,
            best_model_dict=resume_state,
            epoch=epoch - 1,
            global_step=global_step,
            recovery_reason=reason,
        )
        mark_checkpoint_complete(prefix)
        complete = sorted(
            (
                item
                for item in os.listdir(save_model_dir)
                if item.startswith(\"recovery_step_\") and item.endswith(\".complete\")
            ),
            key=lambda item: int(item[len(\"recovery_step_\") : -len(\".complete\")]),
        )
        for stale_marker in complete[:-recovery_keep]:
            stale_prefix = stale_marker[: -len(\".complete\")]
            for suffix in (\".pdparams\", \".pdopt\", \".states\", \".complete\"):
                stale_path = os.path.join(save_model_dir, stale_prefix + suffix)
                if os.path.exists(stale_path):
                    os.remove(stale_path)
        logger.info(\"saved recovery checkpoint %s (%s)\", prefix, reason)
""",
    )
    source = _replace_once(
        source,
        """        for idx, batch in enumerate(train_dataloader):
            model.train()
""",
        """        for idx, batch in enumerate(train_dataloader):
            if epoch == start_epoch and idx < resume_batch:
                reader_start = time.time()
                continue
            model.train()
""",
    )
    source = _replace_once(
        source,
        """            if wd_scheduler is not None:
                wd_scheduler.step()

            # logger and visualdl
""",
        """            if wd_scheduler is not None:
                wd_scheduler.step()
            should_save_recovery = (
                save_batch_step > 0 and global_step % save_batch_step == 0
            )
            if should_save_recovery or stop_state[\"requested\"]:
                reason = \"signal\" if stop_state[\"requested\"] else \"periodic\"
                save_recovery_checkpoint(reason)
                if stop_state[\"requested\"]:
                    logger.warning(\"graceful stop completed at step %s\", global_step)
                    return

            # logger and visualdl
""",
    )
    source = _replace_once(
        source,
        """                    prefix = "best_accuracy"
                    if uniform_output_enabled:
""",
        """                    prefix = "best_accuracy"
                    begin_checkpoint(prefix)
                    if uniform_output_enabled:
""",
    )
    source = _replace_once(
        source,
        """                        global_step=global_step,
                    )
                best_str = "best metric, {}".format(
""",
        """                        global_step=global_step,
                    )
                    mark_checkpoint_complete(prefix)
                best_str = "best metric, {}".format(
""",
    )
    source = _replace_once(
        source,
        """        if dist.get_rank() == 0:
            prefix = "latest"
            # Apply EMA weights for save
""",
        """        if dist.get_rank() == 0:
            prefix = "latest"
            begin_checkpoint(prefix)
            # Apply EMA weights for save
""",
    )
    source = _replace_once(
        source,
        """                global_step=global_step,
            )
            # Restore training weights
""",
        """                global_step=global_step,
            )
            mark_checkpoint_complete(prefix)
            # Restore training weights
""",
    )
    source = _replace_once(
        source,
        """        if dist.get_rank() == 0 and epoch > 0 and epoch % save_epoch_step == 0:
            prefix = "iter_epoch_{}".format(epoch)
            # Apply EMA weights for save
""",
        """        if dist.get_rank() == 0 and epoch > 0 and epoch % save_epoch_step == 0:
            prefix = "iter_epoch_{}".format(epoch)
            begin_checkpoint(prefix)
            # Apply EMA weights for save
""",
    )
    source = _replace_once(
        source,
        """                done_flag=epoch == config["Global"]["epoch_num"],
            )
            # Restore training weights
""",
        """                done_flag=epoch == config["Global"]["epoch_num"],
            )
            mark_checkpoint_complete(prefix)
            # Restore training weights
""",
    )
    source = _replace_once(
        source,
        """            reader_start = time.time()
        if dist.get_rank() == 0:
""",
        """            reader_start = time.time()
        resume_batch = 0
        if dist.get_rank() == 0:
""",
    )
    path.write_text(source, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch pinned PaddleOCR for safe mid-epoch saves")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print("patched" if patch_program(args.path) else "already patched")


if __name__ == "__main__":
    main()
