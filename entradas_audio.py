import os
from mingus.midi import fluidsynth
from mingus.containers import Note

note_sequences = {}
note_sequences['one_note'] = ['C-5']
note_sequences['two_notes'] = ['E-5','G-5']
note_sequences['scale1'] = [ 'C-5','D-5','E-5','F-5','G-5','A-5','B-5']
note_sequences['scale2'] = [ 'C#-5','D#-5','F#-5','G#-5', 'A#-5',]
separations_seconds = [0.5, 1, 2]

fluidsynth.init(sf2='FluidR3 GM2-2.SF2')

def play_sequence(sequence, separation_seconds):
    for note_str in sequence:
        note = Note(note_str)
        fluidsynth.play_Note(note)
        fluidsynth.midi.sleep(seconds=separation_seconds)
        fluidsynth.stop_Note(note)

# Genero audios y los guardo metiendo como entrada las notas con distintas duraciones

temp_n = 1
for sequence in note_sequences:
    for separation_seconds in separations_seconds:
        temp_filename = '/tmp/temp%d.wav' % temp_n
        fluidsynth.midi.start_recording(temp_filename)
        for _ in range(5):
            play_sequence(note_sequences[sequence], separation_seconds)

        final_filename = \
            'audio_inputs/%s_%.1f_s.wav' % (sequence, separation_seconds)
        # colocacion en el canal derecho y mezcla a 10 kHz 
        os.system("sox %s %s remix 1 rate 10000" % \
            (temp_filename, final_filename))
        temp_n += 1

