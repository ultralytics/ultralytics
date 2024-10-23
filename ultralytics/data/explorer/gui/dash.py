# Ultralytics YOLO 🚀, AGPL-3.0 license

import sys
import time
from threading import Thread

from ultralytics import Explorer
from ultralytics.utils import ROOT, SETTINGS
from ultralytics.utils.checks import check_requirements

check_requirements(("streamlit>=1.29.0", "streamlit-select>=0.3"))

import streamlit as st
from streamlit_select import image_select


def _get_explorer():
    """Initializes and returns an instance of the Explorer class."""
    exp = Explorer(data=st.session_state.get("dataset"), model=st.session_state.get("model"))
    thread = Thread(
        target=exp.create_embeddings_table,
        kwargs={"force": st.session_state.get("force_recreate_embeddings"), "split": st.session_state.get("split")},
    )
    thread.start()
    progress_bar = st.progress(0, text="Creating embeddings table...")
    while exp.progress < 1:
        time.sleep(0.1)
        progress_bar.progress(exp.progress, text=f"Progress: {exp.progress * 100}%")
    thread.join()
    st.session_state["explorer"] = exp
    progress_bar.empty()


def init_explorer_form(data=None, model=None):
    """Initializes an Explorer instance and creates embeddings table with progress tracking."""
    if data is None:
        datasets = ROOT / "cfg" / "datasets"
        ds = [d.name for d in datasets.glob("*.yaml")]
    else:
        ds = [data]

    if model is None:
        models = [
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
            "yolov8n-seg.pt",
            "yolov8s-seg.pt",
            "yolov8m-seg.pt",
            "yolov8l-seg.pt",
            "yolov8x-seg.pt",
            "yolov8n-pose.pt",
            "yolov8s-pose.pt",
            "yolov8m-pose.pt",
            "yolov8l-pose.pt",
            "yolov8x-pose.pt",
        ]
    else:
        models = [model]

    splits = ["train", "val", "test"]

    with st.form(key="explorer_init_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.selectbox("Select dataset", ds, key="dataset")
        with col2:
            st.selectbox("Select model", models, key="model")
        with col3:
            st.selectbox("Select split", splits, key="split")
        st.checkbox("Force recreate embeddings", key="force_recreate_embeddings")

        st.form_submit_button("Explore", on_click=_get_explorer)


def query_form():
    """Sets up a form in Streamlit to initialize Explorer with dataset and model selection."""
    with st.form("query_form"):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.text_input(
                "Query",
                "WHERE labels LIKE '%person%' AND labels LIKE '%dog%'",
                label_visibility="collapsed",
                key="query",
            )
        with col2:
            st.form_submit_button("Query", on_click=run_sql_query)


def ai_query_form():
    """Sets up a Streamlit form for user input to initialize Explorer with dataset and model selection."""
    with st.form("ai_query_form"):
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.text_input("Query", "Show images with 1 person and 1 dog", label_visibility="collapsed", key="ai_query")
        with col2:
            st.form_submit_button("Ask AI", on_click=run_ai_query)


def find_similar_imgs(imgs):
    """Initializes a Streamlit form for AI-based image querying with custom input."""
    exp = st.session_state["explorer"]
    similar = exp.get_similar(img=imgs, limit=st.session_state.get("limit"), return_type="arrow")
    paths = similar.to_pydict()["im_file"]
    st.session_state["imgs"] = paths
    st.session_state["res"] = similar


def similarity_form(selected_imgs):
    """Initializes a form for AI-based image querying with custom input in Streamlit."""
    st.write("Similarity Search")
    with st.form("similarity_form"):
        subcol1, subcol2 = st.columns([1, 1])
        with subcol1:
            st.number_input(
                "limit", min_value=None, max_value=None, value=25, label_visibility="collapsed", key="limit"
            )

        with subcol2:
            disabled = not len(selected_imgs)
            st.write("Selected: ", len(selected_imgs))
            st.form_submit_button(
                "Search",
                disabled=disabled,
                on_click=find_similar_imgs,
                args=(selected_imgs,),
            )
        if disabled:
            st.error("Select at least one image to search.")


# def persist_reset_form():
#    with st.form("persist_reset"):
#        col1, col2 = st.columns([1, 1])
#        with col1:
#            st.form_submit_button("Reset", on_click=reset)
#
#        with col2:
#            st.form_submit_button("Persist", on_click=update_state, args=("PERSISTING", True))


def run_sql_query():
    """Executes an SQL query and returns the results."""
    st.session_state["error"] = None
    query = st.session_state.get("query")
    if query.rstrip().lstrip():
        exp = st.session_state["explorer"]
        res = exp.sql_query(query, return_type="arrow")
        st.session_state["imgs"] = res.to_pydict()["im_file"]
        st.session_state["res"] = res


def run_ai_query():
    """Execute SQL query and update session state with query results."""
    if not SETTINGS["openai_api_key"]:
        st.session_state["error"] = (
            'OpenAI API key not found in settings. Please run yolo settings openai_api_key="..."'
        )
        return
    import pandas  # scope for faster 'import ultralytics'

    st.session_state["error"] = None
    query = st.session_state.get("ai_query")
    if query.rstrip().lstrip():
        exp = st.session_state["explorer"]
        res = exp.ask_ai(query)
        if not isinstance(res, pandas.DataFrame) or res.empty:
            st.session_state["error"] = "No results found using AI generated query. Try another query or rerun it."
            return
        st.session_state["imgs"] = res["im_file"].to_list()
        st.session_state["res"] = res


def reset_explorer():
    """Resets the explorer to its initial state by clearing session variables."""
    st.session_state["explorer"] = None
    st.session_state["imgs"] = None
    st.session_state["error"] = None


def utralytics_explorer_docs_callback():
    """Resets the explorer to its initial state by clearing session variables."""
    with st.container(border=True):
        st.image(
            "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg",
            width=100,
        )
        st.markdown(
            "<p>This demo is built using Ultralytics Explorer API. Visit <a href='https://docs.ultralytics.com/datasets/explorer/'>API docs</a> to try examples & learn more</p>",
            unsafe_allow_html=True,
            help=None,
        )
        st.link_button("Ultrlaytics Explorer API", "https://docs.ultralytics.com/datasets/explorer/")


def layout(data=None, model=None):
    """Resets explorer session variables and provides documentation with a link to API docs."""
    st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
    st.markdown("<h1 style='text-align: center;'>Ultralytics Explorer Demo</h1>", unsafe_allow_html=True)

    if st.session_state.get("explorer") is None:
        init_explorer_form(data, model)
        return

    st.button(":arrow_backward: Select Dataset", on_click=reset_explorer)
    exp = st.session_state.get("explorer")
    col1, col2 = st.columns([0.75, 0.25], gap="small")
    imgs = []
    if st.session_state.get("error"):
        st.error(st.session_state["error"])
    elif st.session_state.get("imgs"):
        imgs = st.session_state.get("imgs")
    else:
        imgs = exp.table.to_lance().to_table(columns=["im_file"]).to_pydict()["im_file"]
        st.session_state["res"] = exp.table.to_arrow()
    total_imgs, selected_imgs = len(imgs), []
    with col1:
        subcol1, subcol2, subcol3, subcol4, subcol5 = st.columns(5)
        with subcol1:
            st.write("Max Images Displayed:")
        with subcol2:
            num = st.number_input(
                "Max Images Displayed",
                min_value=0,
                max_value=total_imgs,
                value=min(500, total_imgs),
                key="num_imgs_displayed",
                label_visibility="collapsed",
            )
        with subcol3:
            st.write("Start Index:")
        with subcol4:
            start_idx = st.number_input(
                "Start Index",
                min_value=0,
                max_value=total_imgs,
                value=0,
                key="start_index",
                label_visibility="collapsed",
            )
        with subcol5:
            reset = st.button("Reset", use_container_width=False, key="reset")
            if reset:
                st.session_state["imgs"] = None
                st.experimental_rerun()

        query_form()
        ai_query_form()
        if total_imgs:
            labels, boxes, masks, kpts, classes = None, None, None, None, None
            task = exp.model.task
            if st.session_state.get("display_labels"):
                labels = st.session_state.get("res").to_pydict()["labels"][start_idx : start_idx + num]
                boxes = st.session_state.get("res").to_pydict()["bboxes"][start_idx : start_idx + num]
                masks = st.session_state.get("res").to_pydict()["masks"][start_idx : start_idx + num]
                kpts = st.session_state.get("res").to_pydict()["keypoints"][start_idx : start_idx + num]
                classes = st.session_state.get("res").to_pydict()["cls"][start_idx : start_idx + num]
            imgs_displayed = imgs[start_idx : start_idx + num]
            selected_imgs = image_select(
                f"Total samples: {total_imgs}",
                images=imgs_displayed,
                use_container_width=False,
                # indices=[i for i in range(num)] if select_all else None,
                labels=labels,
                classes=classes,
                bboxes=boxes,
                masks=masks if task == "segment" else None,
                kpts=kpts if task == "pose" else None,
            )

    with col2:
        similarity_form(selected_imgs)
        st.checkbox("Labels", value=False, key="display_labels")
        utralytics_explorer_docs_callback()


if __name__ == "__main__":
    kwargs = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    layout(**kwargs)
