import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
from difflib import SequenceMatcher
from collections import Counter

# --------------------------
# Post-processing functions
# --------------------------
def repetition_score(text):
    """
    Detects repetitive lines in lyrics.
    Returns 0-1 where 1 = good variety, 0 = highly repetitive
    """
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    if len(lines) < 4:
        return 1.0

    def fuzzy_similarity(a, b):
        """Calculate similarity ratio between two strings"""
        return SequenceMatcher(None, a, b).ratio()

    # 1. Exact duplicates
    unique_lines = len(set(lines))
    exact_duplicate_ratio = unique_lines / len(lines)

    # 2. Fuzzy duplicate detection (similarity threshold)
    fuzzy_threshold = 0.85
    fuzzy_duplicates = 0

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if fuzzy_similarity(lines[i], lines[j]) >= fuzzy_threshold:
                fuzzy_duplicates += 1

    max_possible_pairs = len(lines) * (len(lines) - 1) / 2
    fuzzy_duplicate_ratio = 1 - (fuzzy_duplicates / max_possible_pairs) if max_possible_pairs > 0 else 1.0

    # 3. Sequential repetition penalty
    sequential_penalty = 0
    for i in range(len(lines) - 1):
        if fuzzy_similarity(lines[i], lines[i + 1]) >= fuzzy_threshold:
            sequential_penalty += 1

    sequential_score = 1 - (sequential_penalty / (len(lines) - 1)) if len(lines) > 1 else 1.0

    # 4. Most common line frequency
    line_counts = Counter(lines)
    max_count = max(line_counts.values())
    frequency_score = 1 - (max_count / len(lines))

    # Combine scores (weighted)
    final_score = (
        exact_duplicate_ratio * 0.25 +
        fuzzy_duplicate_ratio * 0.25 +
        sequential_score * 0.30 +
        frequency_score * 0.20
    )

    return max(0.0, min(1.0, final_score))


def structure_score(text):
    """Structure score with outlier filtering"""
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    if len(lines) < 4:
        return 0.0

    # Character-level line lengths
    char_lengths = np.array([len(line) for line in lines])

    # Remove outliers (beyond 1.5 std from mean)
    mean_len = np.mean(char_lengths)
    std_len = np.std(char_lengths)

    mask = np.abs(char_lengths - mean_len) <= (1.5 * std_len)
    filtered_lengths = char_lengths[mask]

    # Need at least 3 lines after filtering
    if len(filtered_lengths) < 3:
        filtered_lengths = char_lengths

    # Calculate MAE on filtered lengths
    filtered_mean = np.mean(filtered_lengths)
    mae = np.mean(np.abs(filtered_lengths - filtered_mean))

    normalized_error = mae / filtered_mean if filtered_mean > 0 else 1.0
    score = np.exp(-normalized_error * 3)

    return score


# --------------------------
# Model loading
# --------------------------
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Proper device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    model.to(device)
    return tokenizer, model, device


MODEL_NAMES = {
    "Model A": "bhaveshadhikari/base-og-tuned",
    "Model B": "bhaveshadhikari/syn-tuned",
    "Model C": "bhaveshadhikari/syn-og-tuned"
}

# --------------------------
# Preload all models at startup
# --------------------------
@st.cache_resource
def load_all_models():
    """Preload all models at startup"""
    models = {}
    with st.spinner("🔄 Loading all models... This may take a minute..."):
        for name, model_name in MODEL_NAMES.items():
            tokenizer, model, device = load_model(model_name)
            models[name] = (tokenizer, model, device)
    return models

# Load models at startup
LOADED_MODELS = load_all_models()

st.set_page_config(page_title="GPT-2 Nepali Lyrics Generator Comparison", layout="wide")

st.title("🎶 Nepali Lyrics Generator Comparison")
st.markdown("Compare outputs from **three fine-tuned Nepali text generation models** for the same prompt and settings.")

# Show device info
if torch.cuda.is_available():
    st.info(f"🚀 Running on GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    st.info("🚀 Running on Apple Silicon (MPS)")
else:
    st.warning("⚠️ Running on CPU - consider enabling GPU for faster inference")

# --------------------------
# Input controls
# --------------------------
prompt = st.text_area("🎤 Enter your prompt:", placeholder="उदाहरण: म तिमी बिना ", height=100)

col1, col2, col3, col4 = st.columns(4)
with col1:
    max_new_tokens = st.number_input("Max New Tokens", min_value=10, max_value=300, value=120)
with col2:
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7)
with col3:
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.9)
with col4:
    use_postprocessing = st.checkbox("🔍 Enable Post-processing", value=True,
                                      help="Generate 30 samples and select the best based on repetition and structure scores")

generate = st.button("🎧 Generate Lyrics")

# --------------------------
# Generation Logic
# --------------------------
if generate:
    if prompt.strip():
        st.subheader("Results")

        if use_postprocessing:
            st.info("🔄 Post-processing enabled: Generating 30 samples per model and selecting the best...")

        # Use columns for side-by-side comparison
        cols = st.columns(3)

        for idx, (name, model_name) in enumerate(MODEL_NAMES.items()):
            with cols[idx]:
                st.markdown(f"### 🎵 {name}")
                st.write(f"*{model_name}*")

                with st.spinner(f"Generating lyrics from {name}..."):
                    # Get preloaded model
                    tokenizer, model, device = LOADED_MODELS[name]
                    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

                    if use_postprocessing:
                        # Generate 30 samples with enhanced parameters
                        outputs = model.generate(
                            input_ids,
                            do_sample=True,
                            top_k=20,
                            top_p=0.95,
                            temperature=0.8,
                            repetition_penalty=1.25,
                            eos_token_id=tokenizer.eos_token_id,
                            min_new_tokens=100,
                            max_new_tokens=200,
                            num_return_sequences=30,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                        # Decode all generations
                        samples = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

                        # 1. Filter by repetition score
                        filtered_samples = [
                            text for text in samples if repetition_score(text) > 0.75
                        ]

                        # If none pass the repetition filter, fall back to all samples
                        if not filtered_samples:
                            filtered_samples = samples

                        # 2. Score by structure
                        scored_samples = [(text, structure_score(text)) for text in filtered_samples]

                        # 3. Select the best one
                        generated_text, best_score = max(scored_samples, key=lambda x: x[1])

                        # Show quality metrics
                        rep_score = repetition_score(generated_text)
                        struct_score = structure_score(generated_text)
                        st.caption(f"📊 Quality: Repetition={rep_score:.2f}, Structure={struct_score:.2f}")

                    else:
                        # Standard generation
                        output = model.generate(
                            input_ids,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=True,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

                    lyrics = generated_text.replace("\n", "<br>")

                # Display neatly with preserved line breaks
                bg_colors = ["#fef9f3", "#f5faff", "#f8fdf6"]
                bg = bg_colors[idx % len(bg_colors)]

                st.markdown(
                    f"""
                    <div style="
                        background-color:{bg};
                        color:#222;
                        padding:15px;
                        border-radius:12px;
                        border:1px solid #ddd;
                        font-family:'Courier New', monospace;
                        white-space:pre-line;
                        line-height:1.6;
                    ">
                        {lyrics}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    else:
        st.warning("Please enter a prompt to generate lyrics.")
