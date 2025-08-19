import os
import json
import inspect
from pathlib import Path
from loguru import logger

from mineru.cli.common import prepare_env, read_fn, convert_pdf_bytes_to_bytes_by_pypdfium2
from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.enum_class import MakeMode

def _pick_md_mode() -> "MakeMode":
    # 兼容不同 MinerU 版本
    candidates = ["MM_MD", "MULTIMODAL_MD", "MARKDOWN", "MD"]
    for name in candidates:
        if hasattr(MakeMode, name):
            return getattr(MakeMode, name)
    return list(MakeMode)[0]

def batch_to_markdown(
    input_dir="pdfs",
    output_dir="output_dir",
    lang="ch",
    method="auto",
    start_page=0,
    end_page=None):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffixes = {".pdf", ".png", ".jpg", ".jpeg"}
    paths = [p for p in input_dir.glob("*") if p.suffix.lower() in suffixes]
    if not paths:
        logger.warning(f"目录 {input_dir.resolve()} 中没有可处理的文件（支持: {', '.join(suffixes)}）")
        return

    # 收集字节与语言
    file_names, bytes_list, lang_list = [], [], []
    for p in paths:
        file_names.append(p.stem)
        b = read_fn(str(p))
        if p.suffix.lower() == ".pdf":
            b = convert_pdf_bytes_to_bytes_by_pypdfium2(b, start_page, end_page)
        bytes_list.append(b)
        lang_list.append(lang)

    # 解析（启用公式/表格，更容易出现图片/表格结果）
    infer_results, all_image_lists, all_pdf_docs, lang_list2, ocr_enabled_list = pipeline_doc_analyze(
        bytes_list, lang_list, parse_method=method, formula_enable=True, table_enable=True
    )

    md_mode = _pick_md_mode()
    logger.info(f"使用 MakeMode: {md_mode if isinstance(md_mode, str) else md_mode.name}")

    # 适配不同版本的 result_to_middle_json 是否支持 formula_enable
    supports_formula = "formula_enable" in inspect.signature(pipeline_result_to_middle_json).parameters

    for i, model_list in enumerate(infer_results):
        pdf_file_name = file_names[i]

        # 准备当前文件的输出目录（images 与 md 目录）
        local_image_dir, local_md_dir = prepare_env(str(output_dir), pdf_file_name, method)
        image_writer = FileBasedDataWriter(local_image_dir)  
        md_writer = FileBasedDataWriter(local_md_dir)

        if supports_formula:
            middle_json = pipeline_result_to_middle_json(
                model_list, all_image_lists[i], all_pdf_docs[i],
                image_writer, lang_list2[i], ocr_enabled_list[i], formula_enable=True
            )
        else:
            middle_json = pipeline_result_to_middle_json(
                model_list, all_image_lists[i], all_pdf_docs[i],
                image_writer, lang_list2[i], ocr_enabled_list[i]
            )

        pdf_info = middle_json["pdf_info"]
        image_dir_for_md = os.path.basename(local_image_dir)

        # 生成 Markdown
        md_str = pipeline_union_make(pdf_info, md_mode, image_dir_for_md)
        md_writer.write_string(f"{pdf_file_name}.md", md_str)

        # 输出 content_list.json
        content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir_for_md)
        md_writer.write_string(
            f"{pdf_file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4)
        )

        exts = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
        img_count = sum(1 for n in os.listdir(local_image_dir) if Path(n).suffix.lower() in exts)
        logger.info(f"✅ {pdf_file_name}: 已生成 {pdf_file_name}.md 与 _content_list.json；图片目录 {local_image_dir}（共 {img_count} 张）")


if __name__ == "__main__":
    batch_to_markdown(
        input_dir="pdfs",
        output_dir="output_dir",
        lang="ch",
        method="auto",
        start_page=0,
        end_page=None
    )

