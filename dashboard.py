import streamlit as st
import subprocess
import os
import json
import pandas as pd
import re
import altair as alt
from ultralytics import YOLO
st.set_page_config(page_title="Interaktívna platforma pre YOLO Object Detection", layout="wide")
st.title("Interaktívna platforma pre YOLO Object Detection")
with open("pretrained_info.json", "r") as f:
    pretrained_info = json.load(f)

tab1, tab2, tab3 = st.tabs(["Detekcia v reálnom čase", "Metriky predtrénovaných modelov", "Metriky trénovaných modelov"])


with tab1:
    st.markdown("Táto sekcia umožňuje detekciu objektov v reálnom čase na ľubovolnom nahranom videu na rôznych verziách predtrénovaných modelov.  ")
    st.markdown("Na detekciu objektov v reálnom čase je potrebné nahrať video vo formáte mp4.")
    st.markdown("Modely sú natrénované na datasete COCO, ktorý obsahuje 80 rôznych tried objektov.")
    url = "https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda"
    st.markdown("Kompletný zoznam tried v [COCO datasete](%s)" % url)
    uploaded_file = st.file_uploader("Nahrajte video pre detekciu objektov", type=["mp4"])
    output_video_path = "output_video.mp4"
    
    if "YOLOmodel" not in st.session_state:
        st.session_state["YOLOmodel"] = "YOLO11"
    if "YOLOsize" not in st.session_state:
        st.session_state["YOLOsize"] = "n"
    
    
    YOLOmodel = st.selectbox(
        "Vyberte si verziu YOLO modelu:",
        ['YOLO11', 'YOLOv8', 'YOLOv5'],
        key="YOLOmodel"
    )
    YOLOsize = st.select_slider(
        "Vyberte si velkosť modelu:",
        options=["n", "s", "m", "l", "x"],
        key="YOLOsize"
    )

    
    model_file = f"{st.session_state['YOLOmodel'].lower()}{st.session_state['YOLOsize']}.pt"
    
    st.markdown(f"#### Základné informácie pre model: {model_file}")
    current_stats = pretrained_info.get(model_file, {})
    col1, col2, col3 = st.columns(3)
    with col1:
            st.metric("Vrstvy", current_stats["layers"], delta=None)
    with col2:
            st.metric("Parametre", current_stats["parameters"], delta=None)
    with col3:
            st.metric("GFLOPS", current_stats["GFLOPs"], delta=None)
    
    submit_button = st.button("Spustiť")
    if submit_button and uploaded_file is not None:
        # Ulozi uploadnute video ako temp. file
        input_video_path = f"uploaded_{uploaded_file.name}"
        with open(input_video_path, "wb") as f:
            f.write(uploaded_file.read())
        # Inferencie cez subproces
        command = ["python", "run_inference.py", model_file, input_video_path, output_video_path]
        subprocess.run(command)

        # Zobrazi vystupne video
        if os.path.exists(output_video_path):
            with open(output_video_path, "rb") as video_file:
                video_bytes = video_file.read()
            st.video(video_bytes)
        os.remove(input_video_path)

with tab2:
    
    
    
    st.markdown("Táto sekcia zobrazuje metriky predtrénovaných modelov YOLOv5, YOLOv8 a YOLO11. Každá verzia modelu má rôzne veľkosti (n, s, m, l, x).")
    st.markdown("Tie sú reprezentované ako body v grafoch a sú zoradené podľa veľkosti modelu.")
    st.markdown("Modely sú natrenované spoločnosťou Ultralytics na datasete COCO.")
    url2 = "https://github.com/ultralytics/ultralytics"
    st.markdown("Kompletný zoznam metrik a ich popis nájdete v [Ultralytics GitHub repozitári](%s)" % url2)
    
    with open("metrics.json", "r") as f:
        all_metrics = json.load(f)

        
    def extract_version(model_name):
        match = re.match(r"(YOLOv?\d+)", model_name, re.IGNORECASE)
        return match.group(1) if match else None

    
    def extract_size(model_name):
        match = re.search(r"(n|s|m|l|x)", model_name)
        return match.group(1) if match else ""

    
    size_order = {"n": 1, "s": 2, "m": 3, "l": 4, "x": 5}

    
    yolo_versions = sorted(set(extract_version(m) for m in all_metrics.keys() if extract_version(m)))
    
    yolo_versions_upper = [version.upper() for version in yolo_versions]
    
    selected_version = st.selectbox("Vyberte verziu YOLO:", yolo_versions_upper).lower()
    
    
    filtered_models = {k: v for k, v in all_metrics.items() if extract_version(k) == selected_version}

    if filtered_models:
        # Konverzia na DataFrame
        df = pd.DataFrame(filtered_models).T.reset_index()
        df.rename(columns={"index": "Model"}, inplace=True)

        # Extrakcia veľkosti modelu a zoradenie
        df["Size"] = df["Model"].apply(extract_size)
        df["Size_Order"] = df["Size"].map(size_order)
        df = df.sort_values("Size_Order")

        # Vytvorenie grafov pre každú metriku
        metrics = [col for col in df.columns if col not in ["Model", "Size", "Size_Order"]]
        charts = []

        for metric in metrics:
            metric_df = df[["Model", "Size_Order", metric]].copy()
            metric_df.rename(columns={metric: "Value"}, inplace=True)

            chart = (
                alt.Chart(metric_df)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Model:N", sort=list(df["Model"]), title="Model"),
                    y=alt.Y("Value:Q", title=f"{metric} Value"),
                    tooltip=["Model", "Value"]
                )
                .properties(width=300, height=500, title=f"{metric.capitalize()} Porovnanie")
                .configure_axis(labelFontSize=10, titleFontSize=12)
                .configure_title(fontSize=14,anchor="start", orient="top")
            )
            charts.append(chart)

        # Grafy v 2x2 rozložení
        col1, col2 = st.columns(2)
        with col1:
            st.altair_chart(charts[0], use_container_width=True)
            if len(charts) > 2:
                st.altair_chart(charts[2], use_container_width=True)
        with col2:
            if len(charts) > 1:
                st.altair_chart(charts[1], use_container_width=True)
            if len(charts) > 3:
                st.altair_chart(charts[3], use_container_width=True)

    else:
        st.write("Neboli nájdené žiadne metriky pre vybranú verziu modelu.")
with tab3:
    st.markdown("Táto sekcia zobrazuje metriky trénovaného modelu YOLO11n na troch rôznych datasetoch: VisDrone, COCO a PASCAL VOC.")
    st.markdown("Modely boli natrénované na 10 epochách")
    traindataset =st.selectbox("Vyberte dataset:", ["VisDrone", "COCO", "PASCAL VOC"])
    if traindataset:
        
        csv_path = f"runs/train/{traindataset.lower()}_11n/results.csv"

        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)

            # Definovanie metrik
            metrics = [
                "train/box_loss", "train/cls_loss", "train/dfl_loss",
                "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
                "metrics/mAP50-95(B)", "val/box_loss", "val/cls_loss", "val/dfl_loss"
            ]

            # Vytvorenie grafov pre každú metriku
            charts = []
            for metric in metrics:
                formatted_metric = metric.replace('/', ' : ')
                chart = (
                    alt.Chart(df)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("epoch:Q", title="Epoch"),
                        y=alt.Y(f"{metric}:Q", title=None),
                        tooltip=["epoch", metric]
                    )
                    
                    .properties(width=250, height=250, title=formatted_metric)
                    .configure_title(fontSize=14,anchor="start", orient="top")
                )
                charts.append(chart)

            # Zobrazenie grafov v 2x5 rozložení
            rows = [st.columns(5) for _ in range(2)]
            for i, chart in enumerate(charts):
                row = rows[i // 5]
                with row[i % 5]:
                    st.altair_chart(chart, use_container_width=True)
        else:
            st.write(f"No results found for the {traindataset} dataset.")
    
    with st.expander("Zobraziť dodatočné grafy a krivky"):
        stat1,stat2,stat3 = st.columns(3)
        if traindataset == "VisDrone":
            with stat1:
                st.image("runs/train/visdrone_11n/confusion_matrix_normalized.png", caption="Confusion matrix (normalized)")
            with stat2:
                st.image("runs/train/visdrone_11n/PR_curve.png"\
                         , caption="Precision-Recall Curve")
            with stat3:
                st.image("runs/train/visdrone_11n/F1_curve.png", caption="F1-Score Curve")
            
        if traindataset == "COCO":
            with stat1:
                st.image("runs/train/coco_11n/confusion_matrix_normalized.png", caption="Confusion matrix (normalized)")
            with stat2:
                st.image("runs/train/coco_11n/PR_curve.png", caption="Precision-Recall Curve")
            with stat3:
                st.image("runs/train/coco_11n/F1_curve.png", caption="F1-Score Curve")
            
        if traindataset == "PASCAL VOC":
            with stat1:
                st.image("runs/train/pascal voc_11n/confusion_matrix_normalized.png", caption="Confusion matrix (normalized)")
            with stat2:
                st.image("runs/train/pascal voc_11n/PR_curve.png", caption="Precision-Recall Curve")
            with stat3:
                st.image("runs/train/pascal voc_11n/F1_curve.png", caption="F1-Score Curve")
            
        st.write("Confusion matrix – Matica, ktorá ukazuje počty správnych a nesprávnych predikcií modelu (TP, FP, FN, TN) pre každú triedu.")
        st.write("PR krivka (Precision-Recall) – Graf závislosti presnosti (Precision) od záchytnosti (Recall) pri rôznych prahoch rozhodovania.")
        st.write("F1 krivka – Graf závislosti F1 skóre (harmonický priemer presnosti a záchytnosti) od prahu rozhodovania.")