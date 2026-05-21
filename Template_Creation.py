# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import nbformat as nbf
import os

# 현재 작업 디렉토리에서 모든 Jupyter 노트북 파일을 필터링
notebooks = [file for file in os.listdir(os.getcwd()) if file.endswith('.ipynb') and 
             file not in ['Merging_Notebooks.ipynb', 'empty_page.ipynb', 'Untitled.ipynb', 'Template_Creation.ipynb', 'Template_PY_Creation.ipynb'] and
             not file.startswith('combine') and not file.startswith('template_') and
             not file.startswith('문제풀이')]

for note in notebooks:
    # 노트북 파일을 읽음
    ntbk = nbf.read(note, nbf.NO_CONVERT)
    new_ntbk = nbf.v4.new_notebook()
    new_ntbk['metadata'] = ntbk['metadata']
    
    new_cells = []
    for cell in ntbk.cells:
        if cell.cell_type == "markdown":
            new_cells.append(cell)
        elif cell.cell_type == "code":
            # code 셀의 주석과 함수 정의(def), 클래스 정의(class)만 남기고 실제 코드는 제거
            new_source = []
            for line in cell.source.split('\n'):
                stripped_line = line.strip()
                if stripped_line.startswith("#") or stripped_line.startswith("def ") or stripped_line.startswith("class "):
                    new_source.append(line)
            new_code_cell = nbf.v4.new_code_cell("\n".join(new_source))
            new_cells.append(new_code_cell)
    
    new_ntbk.cells = new_cells
    
    # 새로운 노트북 파일로 저장
    nbf.write(new_ntbk, "template_" + note, version=nbf.NO_CONVERT)


# %%
