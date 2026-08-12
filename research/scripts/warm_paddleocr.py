from __future__ import annotations

from paddleocr import PaddleOCR


def main() -> None:
    common = {
        "lang": "korean",
        "ocr_version": "PP-OCRv5",
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_recognition_model_name": "korean_PP-OCRv5_mobile_rec",
        "use_doc_orientation_classify": False,
        "use_textline_orientation": False,
        "device": "cpu",
        "enable_mkldnn": False,
        "cpu_threads": 4,
        "text_det_limit_side_len": 1536,
        "text_det_limit_type": "max",
        "text_recognition_batch_size": 1,
    }
    PaddleOCR(
        **common,
        use_doc_unwarping=False,
    )
    PaddleOCR(
        **common,
        use_doc_unwarping=True,
    )


if __name__ == "__main__":
    main()
