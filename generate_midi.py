from . import xml_midi_matching as matching
from . import utils
from . import xml_utils
from pyScoreParser.feature_utils import Tempo
import copy


def apply_tempo_perform_features(piece_data, features, start_time=0, predicted=False):
    beats = piece_data.xml_obj.get_beat_positions()
    num_beats = len(beats)
    num_notes = len(piece_data.xml_notes)
    tempos = []
    ornaments = []
    # xml_positions = [x.note_duration.xml_position for x in xml_notes]
    prev_vel = 64
    previous_position = None
    current_sec = start_time
    key_signatures = piece_data.xml_obj.get_key_signatures()
    trill_accidentals = piece_data.xml_obj.get_accidentals

    valid_notes = matching.make_available_note_feature_list(piece_data.xml_notes, features, is_prediction_of_model=predicted)
    previous_tempo = 0

    for i in range(num_beats - 1):
        beat = beats[i]
        feat = utils.get_item_by_xml_position(valid_notes, beat)
        start_position = feat.xml_position
        if start_position == previous_position:
            continue

        if predicted:
            start_index = feat.index
            qpm_saved = 10 ** features['beat_tempo'][start_index]
            num_added = 1
            next_beat = beats[i + 1]

            # calculate average beat_tempo of selected beat
            for j in range(1, start_index): # search backward
                if start_index - j < 0:
                    break
                previous_note = piece_data.xml_notes[start_index - j]
                previous_pos = previous_note.note_duration.xml_position
                if previous_pos == start_position:
                    qpm_saved += 10 ** features['beat_tempo'][start_index - j]
                    num_added += 1
                else:
                    break
            for j in range(1, num_notes-start_index): # search forward
                next_note = piece_data.xml_notes[start_index + j]
                next_position = next_note.note_duration.xml_position
                if next_position < next_beat:
                    qpm_saved += 10 ** features['beat_tempo'][start_index + j]
                    num_added += 1
                else:
                    break
            qpm = qpm_saved / num_added
        else:
            qpm = 10 ** feat.qpm
        # qpm = 10 ** feat.qpm
        divisions = feat.divisions

        if previous_tempo != 0:
            passed_second = (start_position - previous_position) / divisions / previous_tempo * 60
        else:
            passed_second = 0
        current_sec += passed_second
        tempo = Tempo(start_position, qpm, time_position=current_sec, end_xml=0, end_time=0)
        if len(tempos) > 0:
            tempos[-1].end_time = current_sec
            tempos[-1].end_xml = start_position

        tempos.append(tempo)

        previous_position = start_position
        previous_tempo = qpm

    def cal_time_position_with_tempo(note, xml_dev, tempos):
        corresp_tempo = utils.get_item_by_xml_position(tempos, note)
        previous_sec = corresp_tempo.time_position
        passed_duration = note.note_duration.xml_position + xml_dev - corresp_tempo.xml_position
        # passed_duration = note.note_duration.xml_position - corresp_tempo.xml_position
        passed_second = passed_duration / note.state_fixed.divisions / corresp_tempo.qpm * 60

        return previous_sec + passed_second

    for i in range(num_notes):
        note = piece_data.xml_notes[i]
        if not features['onset_deviation'][i] == None:
            xml_deviation = features['onset_deviation'][i] * note.state_fixed.divisions
            # if feat.xml_deviation >= 0:
            #     xml_deviation = (feat.xml_deviation ** 2) * note.state_fixed.divisions
            # else:
            #     xml_deviation = -(feat.xml_deviation ** 2) * note.state_fixed.divisions
        else:
            xml_deviation = 0

        note.note_duration.time_position = cal_time_position_with_tempo(note, xml_deviation, tempos)

        # if not feat['xml_deviation'] == None:
        #     note.note_duration.time_position += feat['xml_deviation']

        end_note = copy.copy(note)
        end_note.note_duration = copy.copy(note.note_duration)
        end_note.note_duration.xml_position = note.note_duration.xml_position + note.note_duration.duration

        end_position = cal_time_position_with_tempo(end_note, 0, tempos)
        if note.note_notations.is_trill:
            note, _ = apply_feat_to_a_note(note, features, i, prev_vel)
            trill_vec = features['trill_param'][i]
            trill_density = trill_vec[0]
            last_velocity = trill_vec[1] * note.velocity
            first_note_ratio = trill_vec[2]
            last_note_ratio = trill_vec[3]
            up_trill = trill_vec[4]
            total_second = end_position - note.note_duration.time_position
            num_trills = int(trill_density * total_second)
            first_velocity = note.velocity

            key = utils.get_item_by_xml_position(key_signatures, note)
            key = key.key
            final_key = None
            for acc in trill_accidentals:
                if acc.xml_position == note.note_duration.xml_position:
                    if acc.type['content'] == '#':
                        final_key = 7
                    elif acc.type['content'] == '♭':
                        final_key = -7
                    elif acc.type['content'] == '♮':
                        final_key = 0
            measure_accidentals = xml_utils.get_measure_accidentals(piece_data.xml_notes, i)
            trill_pitch = note.pitch
            up_pitch, up_pitch_string = xml_utils.cal_up_trill_pitch(note.pitch, key, final_key, measure_accidentals)

            if up_trill:
                up = True
            else:
                up = False

            if num_trills > 2:
                mean_second = total_second / num_trills
                normal_second = (total_second - mean_second * (first_note_ratio + last_note_ratio)) / (num_trills -2)
                prev_end = note.note_duration.time_position
                for j in range(num_trills):
                    if up:
                        pitch = (up_pitch_string, up_pitch)
                        up = False
                    else:
                        pitch = trill_pitch
                        up = True
                    if j == 0:
                        note.pitch = pitch
                        note.note_duration.seconds = mean_second * first_note_ratio
                        prev_end += mean_second * first_note_ratio
                    else:
                        new_note = copy.copy(note)
                        new_note.pedals = None
                        new_note.pitch = copy.copy(note.pitch)
                        new_note.pitch = pitch
                        new_note.note_duration = copy.copy(note.note_duration)
                        new_note.note_duration.time_position = prev_end
                        if j == num_trills - 1:
                            new_note.note_duration.seconds = mean_second * last_note_ratio
                        else:
                            new_note.note_duration.seconds = normal_second
                        new_note.velocity = copy.copy(note.velocity)
                        new_note.velocity = first_velocity + (last_velocity - first_velocity) * (j / num_trills)
                        prev_end += new_note.note_duration.seconds
                        ornaments.append(new_note)
            elif num_trills == 2:
                mean_second = total_second / num_trills
                prev_end = note.note_duration.time_position
                for j in range(2):
                    if up:
                        pitch = (up_pitch_string, up_pitch)
                        up = False
                    else:
                        pitch = trill_pitch
                        up = True
                    if j == 0:
                        note.pitch = pitch
                        note.note_duration.seconds = mean_second * first_note_ratio
                        prev_end += mean_second * first_note_ratio
                    else:
                        new_note = copy.copy(note)
                        new_note.pedals = None
                        new_note.pitch = copy.copy(note.pitch)
                        new_note.pitch = pitch
                        new_note.note_duration = copy.copy(note.note_duration)
                        new_note.note_duration.time_position = prev_end
                        new_note.note_duration.seconds = mean_second * last_note_ratio
                        new_note.velocity = copy.copy(note.velocity)
                        new_note.velocity = last_velocity
                        prev_end += mean_second * last_note_ratio
                        ornaments.append(new_note)
            else:
                note.note_duration.seconds = total_second
        else:
            note.note_duration.seconds = end_position - note.note_duration.time_position

        note, prev_vel = apply_feat_to_a_note(note, features, i, prev_vel)


    # Handle grace notes
    for i in range(num_notes):
        note = piece_data.xml_notes[i]
        if note.note_duration.is_grace_note and note.note_duration.duration == 0:
            for j in range(i+1, num_notes):
                next_note = piece_data.xml_notes[j]
                if not next_note.note_duration.duration == 0 \
                    and next_note.note_duration.xml_position == note.note_duration.xml_position \
                    and next_note.voice == note.voice:
                    next_second = next_note.note_duration.time_position
                    note.note_duration.seconds = (next_second - note.note_duration.time_position) / note.note_duration.num_grace
                    break

    xml_notes = piece_data.xml_notes + ornaments
    xml_notes.sort(key=lambda x: (x.note_duration.xml_position, x.note_duration.time_position, -x.pitch[1]) )
    return xml_notes


def apply_feat_to_a_note(note, features, index, prev_vel):
    if features['articulation'][index] is not None:
        note.note_duration.seconds *= 10 ** (features['articulation'][index])
        # note.note_duration.seconds *= feat.articulation
    if features['velocity'][index] is not None:
        note.velocity = features['velocity'][index]
        prev_vel = note.velocity
    else:
        note.velocity = prev_vel
    if features['pedal_at_start'][index] is not None:
        # note.pedal.at_start = feat['pedal_at_start']
        # note.pedal.at_end = feat['pedal_at_end']
        # note.pedal.refresh = feat['pedal_refresh']
        # note.pedal.refresh_time = feat['pedal_refresh_time']
        # note.pedal.cut = feat['pedal_cut']
        # note.pedal.cut_time = feat['pedal_cut_time']
        # note.pedal.soft = feat['soft_pedal']
        note.pedal.at_start = int(round(features['pedal_at_start'][index]))
        note.pedal.at_end = int(round(features['pedal_at_end'][index]))
        note.pedal.refresh = int(round(features['pedal_refresh'][index]))
        note.pedal.refresh_time = features['pedal_refresh_time'][index]
        note.pedal.cut = int(round(features['pedal_cut'][index]))
        note.pedal.cut_time = features['pedal_cut_time'][index]
        note.pedal.soft = int(round(features['soft_pedal'][index]))
    return note, prev_vel


# TODO: split generate trill code as a single function
def generate_trill(piecd_data, features, i , end_position, prev_vel):
    ornaments = []
    note, _ = apply_feat_to_a_note(piecd_data.xml_notes[i], features, i, prev_vel)
    trill_vec = features['trill_param'][i]
    trill_density = trill_vec[0]
    last_velocity = trill_vec[1] * note.velocity
    first_note_ratio = trill_vec[2]
    last_note_ratio = trill_vec[3]
    up_trill = trill_vec[4]
    total_second = end_position - note.note_duration.time_position
    num_trills = int(trill_density * total_second)
    first_velocity = note.velocity

    key = utils.get_item_by_xml_position(key_signatures, note)
    key = key.key
    final_key = None
    for acc in trill_accidentals:
        if acc.xml_position == note.note_duration.xml_position:
            if acc.type['content'] == '#':
                final_key = 7
            elif acc.type['content'] == '♭':
                final_key = -7
            elif acc.type['content'] == '♮':
                final_key = 0
    measure_accidentals = xml_utils.get_measure_accidentals(piece_data.xml_notes, i)
    trill_pitch = note.pitch
    up_pitch, up_pitch_string = xml_utils.cal_up_trill_pitch(note.pitch, key, final_key, measure_accidentals)

    if up_trill:
        up = True
    else:
        up = False

    if num_trills > 2:
        mean_second = total_second / num_trills
        normal_second = (total_second - mean_second * (first_note_ratio + last_note_ratio)) / (num_trills - 2)
        prev_end = note.note_duration.time_position
        for j in range(num_trills):
            if up:
                pitch = (up_pitch_string, up_pitch)
                up = False
            else:
                pitch = trill_pitch
                up = True
            if j == 0:
                note.pitch = pitch
                note.note_duration.seconds = mean_second * first_note_ratio
                prev_end += mean_second * first_note_ratio
            else:
                new_note = copy.copy(note)
                new_note.pedals = None
                new_note.pitch = copy.copy(note.pitch)
                new_note.pitch = pitch
                new_note.note_duration = copy.copy(note.note_duration)
                new_note.note_duration.time_position = prev_end
                if j == num_trills - 1:
                    new_note.note_duration.seconds = mean_second * last_note_ratio
                else:
                    new_note.note_duration.seconds = normal_second
                new_note.velocity = copy.copy(note.velocity)
                new_note.velocity = first_velocity + (last_velocity - first_velocity) * (j / num_trills)
                prev_end += new_note.note_duration.seconds
                ornaments.append(new_note)
    elif num_trills == 2:
        mean_second = total_second / num_trills
        prev_end = note.note_duration.time_position
        for j in range(2):
            if up:
                pitch = (up_pitch_string, up_pitch)
                up = False
            else:
                pitch = trill_pitch
                up = True
            if j == 0:
                note.pitch = pitch
                note.note_duration.seconds = mean_second * first_note_ratio
                prev_end += mean_second * first_note_ratio
            else:
                new_note = copy.copy(note)
                new_note.pedals = None
                new_note.pitch = copy.copy(note.pitch)
                new_note.pitch = pitch
                new_note.note_duration = copy.copy(note.note_duration)
                new_note.note_duration.time_position = prev_end
                new_note.note_duration.seconds = mean_second * last_note_ratio
                new_note.velocity = copy.copy(note.velocity)
                new_note.velocity = last_velocity
                prev_end += mean_second * last_note_ratio
                ornaments.append(new_note)
    else:
        note.note_duration.seconds = total_second