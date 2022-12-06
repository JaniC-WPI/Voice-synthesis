from pathlib import Path
import src.encoder.inference as encoder
import src.vocoder.inference as vocoder
from src.synthesizer.inference import Synthesizer
import src.vocoder.inference as vocoder
import os
import librosa
import numpy as np
import soundfile as sf
import io


encoder_weights = Path("/home/jc_merlab/voice_cloning_dataset/pretrained/encoder/saved_models/pretrained.pt")
print(encoder_weights)
vocoder_weights = Path("/home/jc_merlab/voice_cloning_dataset/pretrained/vocoder/saved_models/pretrained/pretrained.pt")
print(vocoder_weights)
syn_dir = Path("/home/jc_merlab/voice_cloning_dataset/pretrained/synthesizer/saved_models/logs-pretrained/taco_pretrained")
print(syn_dir)
encoder.load_model(encoder_weights)
print(encoder.is_loaded())
print("is it stopping here")
synthesizer = Synthesizer(syn_dir)
print("or here")
vocoder.load_model(vocoder_weights)
print(vocoder)

def save_audio_local(generated_wav, speaker_name, sample_rate):
    print("is save audio getting called")
    save_dir = 'src\samples\Synthesized_Samples'
    file_path = os.path.join(save_dir, speaker_name + "_synthesized.wav")
    sf.write(file_path, generated_wav, sample_rate, 'PCM_24')

def synthesized_voice(text, speaker_name):
    print("is voice synthesis getting called")
    sample_dir = "src\samples\Original_Samples"
    in_fpath = os.path.join(sample_dir, speaker_name + '.mp3') 
    reprocessed_wav = encoder.preprocess_wav(in_fpath)
    original_wav, sampling_rate = librosa.load(in_fpath)
    preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
    embed = encoder.embed_utterance(preprocessed_wav)
    print("Synthesizing new audio...")
    # with io.capture_output() as captured:
    specs = synthesizer.synthesize_spectrograms([text], [embed])
    generated_wav = vocoder.infer_waveform(specs[0])
    generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
    print("Synthesized audio generated")
    save_audio_local(generated_wav, speaker_name, sampling_rate)
    return generated_wav, synthesizer.sample_rate


    # librosa.output.write_wav(file_path, , sample_rate)

# def main():
#     print("is main getting called")
     

if __name__ == "__main__" :
    print("is main getting called")
    text = "hey little baby, what's up"
    synthesized_voice(text, 'honey_singh')

