import re
from io import StringIO
from pathlib import Path

import emoji
import joblib
# import sklearn # Already imported via sklearn.metrics, sklearn.base
import unicodedata
import numpy as np
import pandas as pd
import streamlit as st

from functools import partial
# import emoji # already imported
from flashtext import KeywordProcessor
from sklearn.base import BaseEstimator, TransformerMixin

from processors.vietnamese_processor import (
    VietnameseToneNormalizer, VietnameseTextCleaner, VietnameseTextPreprocessor, CustomPreprocessorTransformer
)

import hashlib # use file content hashing

# --- Global Definitions and Setup ---
st.set_page_config(
    page_title="ABSA Hotels",
    page_icon="🏨",
    layout="wide",
)

HASHTAG = 'hashtag'

# --- Model and Constants ---
try:
    pipeline_fp = Path('./model/pipe.joblib')
    full_pipeline = joblib.load(pipeline_fp)
except FileNotFoundError:
    st.error("Model file (pipe.joblib) not found. Please ensure it's in the './model/' directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    st.stop()

aspects = [
    'facilities#cleanliness','facilities#comfort','facilities#design&features','facilities#general','facilities#miscellaneous','facilities#prices',
    'facilities#quality','food&drinks#miscellaneous','food&drinks#prices','food&drinks#quality','food&drinks#style&options',
    'hotel#cleanliness','hotel#comfort','hotel#design&features','hotel#general','hotel#miscellaneous','hotel#prices',
    'hotel#quality','location#general','rooms#cleanliness','rooms#comfort','rooms#design&features','rooms#general',
    'rooms#miscellaneous','rooms#prices','rooms#quality','room_amenities#cleanliness','room_amenities#comfort',
    'room_amenities#design&features','room_amenities#general','room_amenities#miscellaneous','room_amenities#prices',
    'room_amenities#quality','service#general'
]

sentiments = ['dne', 'positive', 'negative', 'neutral']
all_keys = aspects

# --- Helper Functions ---
def get_file_hash(file_obj):
    file_content = file_obj.getvalue()
    return hashlib.md5(file_content).hexdigest()

def classify_sentence(sentence_text): # Renamed arg
    return full_pipeline.predict([sentence_text])[0].astype(np.uint)

def display_result(result_array):
    for key, sentiment_idx in zip(all_keys, result_array):
        if not sentiment_idx: # if dne (0)
            continue
        st.markdown(f'- **{key}**: {sentiments[sentiment_idx]}')

def label_encoder(label_str): 
    y_encoded = [np.nan] * len(all_keys)
    ap_stm = re.findall(r'{(.+?)\s*,\s*([a-z]+)}', label_str)
    for aspect, sentiment_str in ap_stm:
        try:
            idx = all_keys.index(aspect)
            if sentiment_str in sentiments: # Ensure sentiment is valid
                y_encoded[idx] = sentiment_str
            else:
                # Optional: Warn about invalid sentiment string
                # st.warning(f"Invalid sentiment '{sentiment_str}' for aspect '{aspect}'. Ignoring.")
                pass
        except ValueError:
            # Optional: Warn about aspect not found
            # st.warning(f"Aspect '{aspect}' in labels not found. Ignoring.")
            pass 
    return y_encoded

def txt2df(uploaded_file_obj):
    stringio = StringIO(uploaded_file_obj.getvalue().decode('utf-8'))
    lines = [line.rstrip('\n\r') for line in stringio.readlines()]

    docs_data = [] 
    current_id_line = None
    active_doc_content_lines = []
    doc_id_pattern = re.compile(r"^#.*")

    for line_content_stripped in lines:
        if doc_id_pattern.match(line_content_stripped):
            if current_id_line is not None: 
                docs_data.append({'id_line': current_id_line, 'content_lines': list(active_doc_content_lines)})
                active_doc_content_lines.clear()
            current_id_line = line_content_stripped
        else:
            if current_id_line is not None: 
                active_doc_content_lines.append(line_content_stripped)
    
    if current_id_line is not None: 
        docs_data.append({'id_line': current_id_line, 'content_lines': list(active_doc_content_lines)})

    if not docs_data and lines: 
        # Heuristic: if first line looks like ID, use it. Otherwise, auto-generate.
        if doc_id_pattern.match(lines[0]):
             docs_data.append({'id_line': lines[0], 'content_lines': lines[1:]})
        else:
             docs_data.append({'id_line': "#1 (auto-generated)", 'content_lines': lines})

    parsed_reviews = []
    parsed_labels_list = [] 
    
    is_globally_labeled = False
    if docs_data and docs_data[0]['content_lines']:
        first_content = docs_data[0]['content_lines']
        if len(first_content) >= 2 and first_content[-1] == '' and \
           re.search(r'\{.*?,\s*.*?\s*\}', first_content[-2]): # Check for label-like pattern
            is_globally_labeled = True

    for doc_entry in docs_data:
        content = doc_entry['content_lines']
        
        if not content:
            parsed_reviews.append("")
            if is_globally_labeled:
                parsed_labels_list.append([np.nan] * len(all_keys))
            continue

        if is_globally_labeled:
            if len(content) >= 2 and content[-1] == '' and re.search(r'\{.*?,\s*.*?\s*\}', content[-2]):
                review_text = "\n".join(content[:-2])
                label_str = content[-2]
                parsed_reviews.append(review_text)
                parsed_labels_list.append(label_encoder(label_str))
            else: 
                review_text = "\n".join(content)
                parsed_reviews.append(review_text)
                parsed_labels_list.append([np.nan] * len(all_keys))
        else: 
            review_text = "\n".join(content)
            parsed_reviews.append(review_text)

    df = pd.DataFrame({'review': parsed_reviews})
    if is_globally_labeled:
        if len(parsed_labels_list) == len(parsed_reviews):
            labels_df = pd.DataFrame(parsed_labels_list, columns=all_keys)
            df = pd.concat([df, labels_df], axis=1)
        else: 
            is_globally_labeled = False 
            for key_col in all_keys: df[key_col] = np.nan
    else:
        for key_col in all_keys: df[key_col] = np.nan
            
    return df, is_globally_labeled

def label_decoder(encoded_label_series): 
    label_parts = []
    for aspect_key, sentiment_val in encoded_label_series.items():
        if pd.notna(sentiment_val) and isinstance(sentiment_val, str) and sentiment_val != 'dne':
            label_parts.append(f'{{{aspect_key}, {sentiment_val}}}')
    return ', '.join(label_parts) if label_parts else "No labels"

def df2txt(df_to_export): 
    X_reviews = df_to_export.review.values
    Y_labels_df = df_to_export[all_keys] 

    rows_output = []
    for test_id, review_text in enumerate(X_reviews, 1):
        label_series = Y_labels_df.iloc[test_id-1]
        decoded_label_str = label_decoder(label_series)
        
        rows_output.append(f'#{test_id}')
        rows_output.append(str(review_text).strip())
        if decoded_label_str and decoded_label_str != "No labels": 
            rows_output.append(decoded_label_str)
            rows_output.append('') 
    return '\n'.join(rows_output)

# --- UI Sections ---

def pre_processing_tool():
    st.header("Pre-processing Tool")
    uploaded_file = st.file_uploader("Upload raw reviews (.txt)", type="txt", key="preprocess_uploader")

    if 'processed_df_for_download' not in st.session_state:
        st.session_state.processed_df_for_download = None

    if uploaded_file:
        raw_df, _ = txt2df(uploaded_file)

        if 'review' not in raw_df.columns or raw_df.empty:
            st.error("Could not parse reviews. Ensure format: #ID\\nReview Text...")
            return

        st.subheader("Original Reviews (first 5)")
        st.dataframe(raw_df[['review']].head())

        text_cleaner_instance = CustomPreprocessorTransformer(use_vncorenlp=True, vncorenlp_dir="./processors/VnCoreNLP")

        if st.button("Pre-process Data", key="preprocess_btn"):
            with st.spinner("Processing..."):
                processed_reviews = text_cleaner_instance.transform(raw_df['review'])
                st.session_state.processed_df_for_download = pd.DataFrame({'review': processed_reviews})
            
            st.subheader("Processed Reviews (first 5)")
            st.dataframe(st.session_state.processed_df_for_download.head())

    if st.session_state.processed_df_for_download is not None and not st.session_state.processed_df_for_download.empty:
        def df_to_txt_unlabeled(df_with_review_col):
            rows = []
            for i, review_text in enumerate(df_with_review_col['review']):
                rows.append(f"#{i+1}")
                rows.append(str(review_text)) 
            return "\n".join(rows)

        txt_data = df_to_txt_unlabeled(st.session_state.processed_df_for_download)
        st.download_button(
            label="Download Pre-processed Data (.txt)",
            data=txt_data,
            file_name="preprocessed_reviews.txt",
            mime="text/plain",
            key="download_preprocessed_btn"
        )

def annotation_tool():
    st.header("Annotation Tool")

    if 'df_annot' not in st.session_state: st.session_state.df_annot = None
    if 'doc_id_annot' not in st.session_state: st.session_state.doc_id_annot = 0
    if 'ndocs_annot' not in st.session_state: st.session_state.ndocs_annot = 0
    if 'prev_fileid_annot' not in st.session_state: st.session_state.prev_fileid_annot = None
    if 'reviewed_docs_annot' not in st.session_state: st.session_state.reviewed_docs_annot = set()

    uploaded_file = st.sidebar.file_uploader('Upload texts to annotate (.txt)', type='txt', key="annotate_uploader")
    
    def refresh_annotation_display():
        state = st.session_state
        if state.df_annot is None or not (0 <= state.doc_id_annot < state.ndocs_annot):
            # Clear radio button states if no valid doc selected or df is None
            for aspect_key_widget in all_keys:
                state[aspect_key_widget] = 'dne'
            return
        
        doc_id = state.doc_id_annot
        current_doc_data = state.df_annot.loc[doc_id]
        
        for aspect_key_widget in all_keys:
            sentiment_val = current_doc_data.get(aspect_key_widget, np.nan)
            if pd.isna(sentiment_val) or sentiment_val == 'dne':
                state[aspect_key_widget] = 'dne'
            else:
                state[aspect_key_widget] = str(sentiment_val)

    if uploaded_file:
        current_file_hash = get_file_hash(uploaded_file)
        if st.session_state.prev_fileid_annot != current_file_hash:
            df_loaded, _ = txt2df(uploaded_file)
            if not df_loaded.empty:
                st.session_state.df_annot = df_loaded
                st.session_state.ndocs_annot = len(df_loaded)
                st.session_state.doc_id_annot = 0 
                st.session_state.prev_fileid_annot = current_file_hash
                st.session_state.reviewed_docs_annot.clear()
                refresh_annotation_display()
            else: # File was empty or unparseable by txt2df
                st.session_state.df_annot = None
                st.session_state.ndocs_annot = 0
                st.warning("Uploaded file is empty or could not be parsed into reviews.")
    elif st.session_state.df_annot is not None and uploaded_file is None: 
        st.session_state.df_annot = None 
        st.session_state.ndocs_annot = 0
        st.session_state.reviewed_docs_annot.clear()
        refresh_annotation_display() # Clear radio buttons


    def on_sentiment_change_annot(aspect_key_changed):
        state = st.session_state
        if state.df_annot is None or not (0 <= state.doc_id_annot < state.ndocs_annot): return
        doc_id = state.doc_id_annot
        new_sentiment = state[aspect_key_changed]
        
        if aspect_key_changed in state.df_annot.columns:
            state.df_annot.loc[doc_id, aspect_key_changed] = new_sentiment if new_sentiment != 'dne' else np.nan

    def navigate_doc_annot(go_next):
        state = st.session_state
        if state.df_annot is None or state.ndocs_annot == 0 : return

        current_doc_id = state.doc_id_annot
        state.reviewed_docs_annot.add(current_doc_id)

        if go_next:
            state.doc_id_annot = min(state.ndocs_annot - 1, current_doc_id + 1)
        else:
            state.doc_id_annot = max(0, current_doc_id - 1)
        refresh_annotation_display()

    def jump_to_doc_annot(target_doc_id):
        state = st.session_state
        if state.df_annot is None or not (0 <= target_doc_id < state.ndocs_annot): return
        state.reviewed_docs_annot.add(state.doc_id_annot)
        state.doc_id_annot = target_doc_id
        refresh_annotation_display()

    def auto_annotate_current_doc():
        state = st.session_state
        if state.df_annot is None or not (0 <= state.doc_id_annot < state.ndocs_annot): return
        doc_id = state.doc_id_annot
        review_text = state.df_annot.loc[doc_id, 'review']
        
        predicted_sentiments_indices = classify_sentence(review_text)
        
        for i, aspect_key_model in enumerate(all_keys):
            sentiment_idx = predicted_sentiments_indices[i]
            state.df_annot.loc[doc_id, aspect_key_model] = sentiments[sentiment_idx] if sentiment_idx != 0 else np.nan
        refresh_annotation_display()

    def auto_annotate_all_docs():
        state = st.session_state
        if state.df_annot is None or state.df_annot.empty: return

        reviews_to_predict = state.df_annot['review']
        all_predictions_indices = full_pipeline.predict(reviews_to_predict)
        
        for doc_idx in range(len(state.df_annot)):
            for aspect_idx, aspect_key_model in enumerate(all_keys):
                sentiment_idx = all_predictions_indices[doc_idx, aspect_idx]
                state.df_annot.loc[doc_idx, aspect_key_model] = sentiments[sentiment_idx] if sentiment_idx != 0 else np.nan
        refresh_annotation_display() # Refresh for current doc


    if st.session_state.df_annot is not None and not st.session_state.df_annot.empty:
        state = st.session_state
        doc_id = state.doc_id_annot 
        
        st.sidebar.subheader("Annotation History")
        reviewed_docs_sorted_annot = sorted(list(state.reviewed_docs_annot))
        for rev_doc_idx in reviewed_docs_sorted_annot:
            if st.sidebar.button(f"Doc #{rev_doc_idx + 1}", key=f"hist_annot_{rev_doc_idx}"):
                jump_to_doc_annot(rev_doc_idx)
        if not reviewed_docs_sorted_annot: st.sidebar.write("No docs visited yet.")

        st.selectbox(
            label='Choose document to annotate:',
            options=range(state.ndocs_annot),
            format_func=lambda x: f"Document {x + 1} / {state.ndocs_annot}",
            key='doc_id_annot',
            on_change=refresh_annotation_display 
        )
        
        current_doc_content = state.df_annot.loc[doc_id, 'review']
        st.text_area("Review Text:", value=current_doc_content, height=150, disabled=True, key=f"review_disp_{doc_id}")
        
        current_labels_series = state.df_annot.loc[doc_id, all_keys]
        st.write(f"**Current Labels:** {label_decoder(current_labels_series)}")

        cols_buttons = st.columns(4)
        cols_buttons[0].button('⏮ Prev Doc', on_click=navigate_doc_annot, args=(False,), key="prev_doc_annot")
        cols_buttons[1].button('Next Doc ⏭', on_click=navigate_doc_annot, args=(True,), key="next_doc_annot")
        cols_buttons[2].button('Auto (This Doc) 🐢', on_click=auto_annotate_current_doc, key="auto_this_annot")
        cols_buttons[3].button('Auto (All Docs) ⚡', on_click=auto_annotate_all_docs, key="auto_all_annot")

        ui_entities_structure = {
            # Nhóm 1: Hotel
            'hotel': [ 
                "hotel#general", "hotel#prices", "hotel#design&features", "hotel#cleanliness", "hotel#comfort", "hotel#quality", "hotel#miscellaneous"
            ],
            # Nhóm 2: Rooms
            'rooms': [
                "rooms#general", "rooms#prices", "rooms#design&features", "rooms#cleanliness", "rooms#comfort", "rooms#quality", "rooms#miscellaneous"
            ],
            # Nhóm 3: Room Amenities
            'room_amenities': [
                "room_amenities#general", "room_amenities#prices", "room_amenities#design&features", "room_amenities#cleanliness", "room_amenities#comfort", "room_amenities#quality", "room_amenities#miscellaneous"
            ],
             # Nhóm 4: Facilities
            'facilities': [
                "facilities#general", "facilities#prices", "facilities#design&features", "facilities#cleanliness", "facilities#comfort", "facilities#quality", "facilities#miscellaneous"
            ],           
            
            # Nhóm 5: Food & Drinks
            'food&drinks': [
                "food&drinks#prices", "food&drinks#quality", "food&drinks#style&options", "food&drinks#miscellaneous"
            ],
            # Nhóm 6: OTHERS (Service, Location)
            'OTHERS (Service, Location)': [
                "service#general", "location#general"
            ]}
        
        for entity_name, aspect_keys_for_entity in ui_entities_structure.items():
            with st.expander(entity_name, expanded=True):
                num_aspects_in_entity = len(aspect_keys_for_entity)
                if num_aspects_in_entity == 0: continue
                cols_aspects = st.columns(num_aspects_in_entity)
                
                for i, aspect_key_radio in enumerate(aspect_keys_for_entity):
                    state.setdefault(aspect_key_radio, 'dne')

                    with cols_aspects[i]:
                        attr_label = aspect_key_radio.split('#')[-1]
                        st.radio(
                            label=attr_label, options=sentiments, key=aspect_key_radio,
                            on_change=on_sentiment_change_annot, args=(aspect_key_radio,)
                        )
        
        st.subheader("Annotation Data Table")
        st.dataframe(state.df_annot)

        if not state.df_annot.empty:
            txt_to_export = df2txt(state.df_annot)
            st.sidebar.download_button(
                label='Export Annotated Data ⬇', data=txt_to_export,
                file_name='annotated_reviews.txt', mime='text/plain', key='export_annot_data_btn'
            )
    else:
        st.info('Upload a text file (UTF-8 encoded) to begin annotation. Format: #ID Review text...')


# --- Main App Navigation ---
st.title('ABSA for Hotel Reviews 🏨')

app_modes = {
    "Classify Single Sentence": None, 
    "Pre-processing Tool": pre_processing_tool,
    "Annotation Tool": annotation_tool
}

st.sidebar.title("Tools")
selected_mode_key = st.sidebar.radio("Select a tool:", list(app_modes.keys()), key='app_mode_selector')

if selected_mode_key == "Classify Single Sentence":
    st.header("Classify Single Sentence")
    sentence_to_classify = st.text_area("Enter review here:", key="classify_sentence_input")
    if st.button("Classify Sentence", key="classify_btn") and sentence_to_classify:
        with st.spinner("Classifying..."):
            classification_result = classify_sentence(sentence_to_classify)
            display_result(classification_result)
elif selected_mode_key in app_modes and app_modes[selected_mode_key] is not None:
    app_modes[selected_mode_key]()
else:
    st.info("Select a tool from the sidebar.")
