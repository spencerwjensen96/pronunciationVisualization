import torch
import torchaudio
import torchaudio.transforms as T
import streamlit as st
import tempfile
import os
import math

from dataclasses import dataclass
import matplotlib.pyplot as plt

from formants import get_formants 

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        if self.label == "|":
            return f"sil [{self.start}, {self.end})"
        return f"{self.label} [{self.start}, {self.end})"

    @property
    def length(self):
        return self.end - self.start


def plot_alignment():
    fig, ax = plt.subplots()
    img = ax.imshow(emission.T)
    ax.set_title("Frame-wise class probability")
    ax.set_xlabel("Time")
    ax.set_ylabel("Labels")
    fig.colorbar(img, ax=ax, shrink=0.6, location="bottom")
    
    fig.tight_layout()
    return fig



def get_trellis(emission, tokens, blank_id=0):
    num_frame = emission.size(0)
    num_tokens = len(tokens)

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis

def plot_trellis():
    fig, ax = plt.subplots()
    img = ax.imshow(trellis.T, origin="lower")
    ax.annotate("- Inf", (trellis.size(1) / 5, trellis.size(1) / 1.5))
    ax.annotate("+ Inf", (trellis.size(0) - trellis.size(1) / 5, trellis.size(1) / 3))
    fig.colorbar(img, ax=ax, shrink=0.6, location="bottom")
    fig.tight_layout()
    return fig


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]

def plot_trellis_with_path(trellis, path):
    # To plot trellis with path, we take advantage of 'nan' value
    trellis_with_path = trellis.clone()
    for _, p in enumerate(path):
        trellis_with_path[p.time_index, p.token_index] = float("nan")
    plt.imshow(trellis_with_path.T, origin="lower")
    plt.title("The path found by backtracking")
    plt.tight_layout()
    return plt

def merge_repeats(path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2
    return segments

def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

def scatter_plot(x, y, x_label, y_label, title):
    plt.scatter(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    return plt

def scatter_plot_multi(x, y, labels, x_label, y_label, title):
    for i, values in enumerate(x):
        plt.scatter(values, y[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(labels)
    return plt

torch.random.manual_seed(0)
audio = st.file_uploader("Upload a file", type=["wav"])
# audio = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")

with st.spinner("Loading model..."):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # st.text(f"Using device: {device}")
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()

if audio is None:
    st.error("Need an audio file!")
else:
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, audio.name)
    with open(audio_path, "wb") as f:
        f.write(audio.getvalue())

    with torch.inference_mode():
        waveform, _ = torchaudio.load(audio_path)
        emissions, _ = model(waveform.to(device))
        emissions = torch.log_softmax(emissions, dim=-1)
        # spectrogram = T.Spectrogram(n_fft=512)
        # spec = spectrogram(waveform)

    emission = emissions[0].cpu().detach()

    st.audio(audio_path)

    input_transcript = st.text_input("Transcript of Audio")
    input_transcript = input_transcript.upper()
    input_transcript = input_transcript.replace(" ", "|")
    input_transcript = "|" + input_transcript + "|"

    transcript = "|FOR|THE|TWENTIETH|TIME|THAT|EVENING|THE|TWO|MEN|SHOOK|HANDS|"

    dictionary = {c: i for i, c in enumerate(labels)}

    if transcript is None:
        st.error("Need a transcript!")

    tokens = [dictionary[c] for c in transcript]
    trellis = get_trellis(emission, tokens)
    path = backtrack(trellis, emission, tokens)
    segments = merge_repeats(path)
    word_segments = merge_words(segments)

    if st.button("Plot Alignment"):
        st.pyplot(plot_alignment())

        for word in word_segments:
            st.text(word)

    if st.button("Plot Trellis"):
        st.pyplot(plot_trellis())

    if st.button("Backtrace"):
        st.pyplot(plot_trellis_with_path(trellis, path))

    st.title("Formant Analysis")
    options = st.multiselect(
        'Which tokens are you interested in?',
        set(transcript))
    ratio = waveform.size(1) / trellis.size(0)
    if st.button("Analyse Formants"):
        f1_values = []
        f2_values = []
        scatter_labels = []
        
        for option in options:
            frames_of_interest = []
            for seg in segments:
                if seg.label == option:
                    start = int(ratio * seg.start)
                    end = int(ratio * seg.end)
                    range_seg = [*range(start, end, 1)] 
                    frames_of_interest += range_seg
            
            metadata = torchaudio.info(audio_path)
            f1, f2 = get_formants(audio_path)
            diff_frames = metadata.num_frames - len(f1)
            additional_frames = [0]*math.ceil(diff_frames/2)
            f1 = additional_frames + f1 + additional_frames
            f2 = additional_frames + f2 + additional_frames

            f1_frames = [f1[i] for i in frames_of_interest]
            f2_frames = [f2[i] for i in frames_of_interest]
            f1_values.append(f1_frames)
            f2_values.append(f2_frames)
            scatter_labels.append(option)
        
        st.pyplot(scatter_plot_multi(f1_values, f2_values, scatter_labels, "F1", "F2", "Formants"))


