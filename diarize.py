import logging
import os
import re

import faster_whisper
import torch
import torchaudio

from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from helpers import (
    cleanup,
    create_config,
    find_numeral_symbol_tokens,
    get_realigned_ws_mapping_with_punctuation,
    get_sentences_speaker_mapping,
    get_speaker_aware_transcript,
    get_words_speaker_mapping,
    langs_to_iso,
    punct_model_langs,
    whisper_langs,
    write_srt,
)

class AudioProcessor:
    def __init__(self, whisper_model_name="medium.en", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.mtypes = {"cpu": "int8", "cuda": "float16"}

        # Load Whisper model
        self.whisper_model = faster_whisper.WhisperModel(
            whisper_model_name, device=self.device, compute_type=self.mtypes[self.device]
        )
        self.whisper_pipeline = faster_whisper.BatchedInferencePipeline(self.whisper_model)

        # Load alignment model
        self.alignment_model, self.alignment_tokenizer = load_alignment_model(
            self.device, dtype=torch.float16 if self.device == "cuda" else torch.float32
        )

    def process_audio(
        self,
        audio_path,
        stemming=True,
        suppress_numerals=False,
        batch_size=8,
        language=None,
    ):
        language = process_language_arg(language, self.whisper_model.name)
        if stemming:
            return_code = os.system(
                f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{audio_path}" -o temp_outputs'
            )

            if return_code != 0:
                logging.warning(
                    "Source splitting failed, using original audio file. "
                    "Use stemming=False to disable it."
                )
                vocal_target = audio_path
            else:
                vocal_target = os.path.join(
                    "temp_outputs",
                    "htdemucs",
                    os.path.splitext(os.path.basename(audio_path))[0],
                    "vocals.wav",
                )
        else:
            vocal_target = audio_path

        # Transcribe the audio file
        audio_waveform = faster_whisper.decode_audio(vocal_target)
        suppress_tokens = (
            find_numeral_symbol_tokens(self.whisper_model.hf_tokenizer)
            if suppress_numerals
            else [-1]
        )

        if batch_size > 0:
            transcript_segments, info = self.whisper_pipeline.transcribe(
                audio_waveform,
                language,
                suppress_tokens=suppress_tokens,
                batch_size=batch_size,
            )
        else:
            transcript_segments, info = self.whisper_model.transcribe(
                audio_waveform,
                language,
                suppress_tokens=suppress_tokens,
                vad_filter=True,
            )

        full_transcript = "".join(segment.text for segment in transcript_segments)

        emissions, stride = generate_emissions(
            self.alignment_model,
            torch.from_numpy(audio_waveform)
            .to(self.alignment_model.dtype)
            .to(self.alignment_model.device),
            batch_size=batch_size,
        )

        tokens_starred, text_starred = preprocess_text(
            full_transcript,
            romanize=True,
            language=langs_to_iso[info.language],
        )
        segments, scores, blank_token = get_alignments(
            emissions,
            tokens_starred,
            self.alignment_tokenizer,
        )
        spans = get_spans(tokens_starred, segments, blank_token)
        word_timestamps = postprocess_results(text_starred, spans, stride, scores)

        # Convert audio to mono for NeMo compatibility
        ROOT = os.getcwd()
        temp_path = os.path.join(ROOT, "temp_outputs")
        os.makedirs(temp_path, exist_ok=True)
        torchaudio.save(
            os.path.join(temp_path, "mono_file.wav"),
            torch.from_numpy(audio_waveform).unsqueeze(0).float(),
            16000,
            channels_first=True,
        )

        # Initialize NeMo MSDD diarization model
        msdd_model = NeuralDiarizer(cfg=create_config(temp_path)).to(self.device)
        msdd_model.diarize()

        del msdd_model
        torch.cuda.empty_cache()

        # Reading timestamps <> Speaker Labels mapping
        speaker_ts = []
        with open(os.path.join(temp_path, "pred_rttms", "mono_file.rttm"), "r") as f:
            lines = f.readlines()
            for line in lines:
                line_list = line.split(" ")
                s = int(float(line_list[5]) * 1000)
                e = s + int(float(line_list[8]) * 1000)
                speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

        wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

        if info.language in punct_model_langs:
            punct_model = PunctuationModel(model="kredor/punctuate-all")
            words_list = list(map(lambda x: x["word"], wsm))
            labled_words = punct_model.predict(words_list, chunk_size=230)
            ending_puncts = ".?!"
            model_puncts = ".,;:!?"

            is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)
            for word_dict, labeled_tuple in zip(wsm, labled_words):
                word = word_dict["word"]
                if (
                    word
                    and labeled_tuple[1] in ending_puncts
                    and (word[-1] not in model_puncts or is_acronym(word))
                ):
                    word += labeled_tuple[1]
                    if word.endswith(".."):
                        word = word.rstrip(".")
                    word_dict["word"] = word
        else:
            logging.warning(
                f"Punctuation restoration is not available for {info.language} language."
                " Using the original punctuation."
            )

        wsm = get_realigned_ws_mapping_with_punctuation(wsm)
        ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

        output_txt = f"{os.path.splitext(audio_path)[0]}.txt"
        output_srt = f"{os.path.splitext(audio_path)[0]}.srt"
        with open(output_txt, "w", encoding="utf-8-sig") as f:
            get_speaker_aware_transcript(ssm, f)
        with open(output_srt, "w", encoding="utf-8-sig") as srt:
            write_srt(ssm, srt)

        cleanup(temp_path)
        return output_txt, output_srt
