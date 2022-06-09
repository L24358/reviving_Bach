import os
import torch
import numpy as np
import music21 as m21
from collections import Counter

def get_midis(filepath):
    '''
    Parameters
    ----------
    filepath : string
        path that contains midi files.

    Returns
    -------
    all_midis : list
        list containing music21-processed scores.
    fnames : TYPE
        filenames corresponding to the scores in all_midis.

    '''
    all_midis, fnames = [], []
    for f in os.listdir(filepath):
        if f.endswith(".mid"):
            fnames.append(f)
            tr = os.path.join(filepath, f)
            midi = m21.converter.parse(tr)
            all_midis.append(midi)
    return all_midis, fnames

def extract_notes(file):
    '''
    Parameters
    ----------
    file : string
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    dic : TYPE
        DESCRIPTION.

    '''
    songs = m21.instrument.partitionByInstrument(file)
    
    dic = {}; all_notes = []
    for part in songs.parts:
        notes = []
        pick = part.recurse()
        for element in pick:
            if isinstance(element, m21.note.Note):
                notes.append([str(element.pitch), float(element.offset)]) #, float(element.duration.quarterLength)
                all_notes.append(str(element.pitch))
            elif isinstance(element, m21.chord.Chord):
                notes.append([str(element.pitches[0]), float(element.offset)]) #".".join(str(n) for n in element.normalOrder)
                all_notes.append(str(element.pitches[0]))
        if part.id in dic.keys(): print('Duplicate instrument detected!')
        dic[part.id] = notes
    return list(set(all_notes)), dic

def extract_notes_batch(files):
    all_notes, all_dic = [], {}
    for file in files:
        notes, dic = extract_notes(file)
        for instrument in dic.keys():
            if instrument not in all_dic.keys(): all_dic[instrument] = []
            all_dic[instrument].append(dic[instrument])
        all_notes += notes
    return all_notes, all_dic

def chords_n_notes(Snippet, Offset, base):
    Melody = []
    for s in range(len(Snippet)):
        i = Snippet[s]
        if ("." in i or i.isdigit()): # Is a chord
            chord_notes = i.split(".") # Seperating the notes in chord
            notes = [] 
            for j in chord_notes:
                inst_note = int(j)
                note_snip = m21.note.Note(inst_note)         
                notes.append(note_snip)
                chord_snip = m21.chord.Chord(notes)
                Melody.append(chord_snip)
        else: # Is a note
            note_snip = m21.note.Note(i); note_snip.offset = Offset[s]
            Melody.append(note_snip)
    Melody_midi = m21.stream.Stream(Melody)   
    return Melody_midi

def remove_rare(Corpus, threshold):
    rare_note = []
    count_num = Counter(Corpus)
    for index, (key, value) in enumerate(count_num.items()):
        if value < threshold:
            m =  key
            rare_note.append(m)
    
    for element in Corpus:
        if element in rare_note: Corpus.remove(element)
    return Corpus, rare_note

def remove_short(instru2corpus, thre=19):
    for instrument in instru2corpus.keys():
        for song in instru2corpus[instrument]:
            if len(song) <= thre: instru2corpus[instrument].remove(song)
    return instru2corpus

def get_map(Corpus):
    order = {'C':1,'D':2,'E':3,'F':4,'G':5,'A':6,'B':7, '#':0.5, '-':-0.5}
    dic, fmap, rmap = {}, {}, {}
    for elem in list(set(Corpus)):
        if not elem[0].isnumeric(): # is note
            if len(elem) == 2:
                note, octave = elem; pitch = order[note]
            else:
                note, black, octave = elem; pitch = order[note] + order[black]
            dic[(int(octave), pitch)] = elem
            
    for idx, key in enumerate(sorted(dic)):
        fmap[dic[key]] = idx  
        rmap[idx] = dic[key]
    return Corpus, fmap, rmap

def breakeven(song_list, song_len, stride):
    new_songlist = []
    for song in song_list:
        t = 0; T = len(song)
        while t+song_len < T:
            new_songlist.append(song[t:t+song_len])
            t += stride
    return new_songlist

def difference(song):
    res = []
    for t in range(len(song)-1):
        res.append([song[t][0], round(song[t+1][1]-song[t][1],2)])
    res.append([song[-1][0], 1])
    return res

def center(song):
    res = []
    t_min = song[0][1]
    for t in range(len(song)):
        res.append([song[t][0], song[t][1]-t_min])
    return res 

def train_test_split(instru2corpus, instrument, fmap, song_len, stride,\
                     seed=None, train_percentage=0.8, process=None):
    if seed!=None: np.random.seed(seed)
    song_list = list(filter(([]).__ne__, instru2corpus[instrument])) # remove empty songs
    song_list = breakeven(song_list, song_len, stride) # break songs into parts of equal length
    if process=='difference':
        song_list = [difference(song) for song in song_list]
    elif process=='center':
        song_list = [center(song) for song in song_list]
    
    X_melody = [[fmap[elem[0]] for elem in song] for song in song_list]
    X_offset = [[elem[1] for elem in song] for song in song_list]
    
    train_indices = np.random.choice(range(len(X_melody)), int(len(X_melody)*train_percentage), replace=False)
    val_indices = [idx for idx in range(len(X_melody)) if idx not in train_indices]
    
    X_train_melody = np.array(X_melody)[train_indices]
    X_val_melody = np.array(X_melody)[val_indices]
    X_train_offset = np.array(X_offset)[train_indices]
    X_val_offset = np.array(X_offset)[val_indices]
    return X_train_melody, X_val_melody, X_train_offset, X_val_offset

def train_test_split_joint(instru2corpus, instrument, fmap, song_len, stride,\
                           seed=None, train_percentage=0.8, process=None):
    if seed!=None: np.random.seed(seed)
    song_list = list(filter(([]).__ne__, instru2corpus[instrument])) # remove empty songs
    song_list = breakeven(song_list, song_len, stride) # break songs into parts of equal length
    if process=='difference':
        song_list = [difference(song) for song in song_list]
    elif process=='center':
        song_list = [center(song) for song in song_list]
    
    X = [[[fmap[elem[0]], elem[1]] for elem in song] for song in song_list]
    train_indices = np.random.choice(range(len(X)), int(len(X)*train_percentage), replace=False)
    val_indices = [idx for idx in range(len(X)) if idx not in train_indices]
    X_train = np.array(X)[train_indices]
    X_val = np.array(X)[val_indices]
    return X_train, X_val

def batchify(X):
    return [torch.from_numpy(np.array(song).reshape(1,1,-1)).float() for song in X]

def batchify_joint(X):
    res = []
    for song in X:
        x = torch.from_numpy(np.array(song).reshape(-1,2).T).float()
        x = x.unsqueeze(0)
        res.append(x)
    return res

def upsample(song, base=12):
    song = difference(song)
    
    res = []
    for note, diff in song:
        rp = int(diff*base)
        res += [[note, 0] for i in range(rp)]
    return res

def upsample_batch(instru2corpus):
    new_instru2corpus = {}
    for instrument in instru2corpus.keys():
        new_instru2corpus[instrument] = []
        song_list = list(filter(([]).__ne__, instru2corpus[instrument]))
        for song in song_list:
            new_instru2corpus[instrument].append(upsample(song))
    return new_instru2corpus

def reconstruct_melody(model1, model2, X_train_melody, X_train_offset, rmap, base=1, idx=None, filename='recon'):
    model1.eval()

    # set song index
    if idx == None: idx = np.random.randint(0, len(X_train_melody))
    
    recons_train_melody = model1(X_train_melody[idx])[0]
    recons_train_melody = [rmap[int(elem)] for elem in recons_train_melody.view(-1).detach().numpy()]
    ori_train_melody = [rmap[int(elem)] for elem in X_train_melody[idx].view(-1).numpy()]
    
    const_offset = np.arange(len(recons_train_melody))
    ori_train_offset = X_train_offset[idx].view(-1).numpy()
    if model2 != None:
        model2.eval()
        recons_train_offset = model2(X_train_offset[idx])[0].view(-1).detach().numpy()
        
        melody_train = chords_n_notes(recons_train_melody, recons_train_offset, base)
        melody_train = m21.stream.Stream(melody_train) 
        melody_train.write('midi', os.path.join('./music/',filename+'_fullrecon_'+str(idx)+'.mid'))
        
    melody_train = chords_n_notes(recons_train_melody, ori_train_offset, base)
    melody_train = m21.stream.Stream(melody_train) 
    melody_train.write('midi', os.path.join('./music/',filename+'_cheat_'+str(idx)+'.mid'))

    melody_train = chords_n_notes(ori_train_melody, ori_train_offset, base)
    melody_train = m21.stream.Stream(melody_train) 
    melody_train.write('midi', os.path.join('./music/',filename+'_original_'+str(idx)+'.mid'))
    
    melody_train = chords_n_notes(recons_train_melody, const_offset, base)
    melody_train = m21.stream.Stream(melody_train) 
    melody_train.write('midi', os.path.join('./music/',filename+'_constrecon_'+str(idx)+'.mid'))

    melody_train = chords_n_notes(ori_train_melody, const_offset, base)
    melody_train = m21.stream.Stream(melody_train) 
    melody_train.write('midi', os.path.join('./music/',filename+'_constori_'+str(idx)+'.mid'))
    
def get_map_offset(X):
    X_np = np.array([item.numpy() for item in X])
    Corpus = sorted(list(set(X_np.reshape(-1))))
    fmap, rmap, count = {}, {}, 0
    for off in Corpus:
        fmap[round(float(off),2)] = count
        rmap[count] = off
        count += 1
    return Corpus, fmap, rmap

def map_offset(X, fmap,song_len):
    new_X = []
    for song in X:
        shape = song.shape
        offs = torch.from_numpy(np.array([fmap[round(off.item(),2)] for off in song.view(-1)])).view(*shape)
        new_X.append(offs)
    new_X = [song.float() for song in new_X]
    return new_X

def safe_map(rmap, melody):
    notes = []
    maxnum = max(rmap.keys())
    for num in melody:
        if (num <= maxnum) and (num > 0): notes.append(rmap[int(num)])
        elif num > maxnum: notes.append(rmap[maxnum])
        else: notes.append(rmap[0])
    return np.array(notes)

def gen_melody(mu_avg, cov_avg, model, rmap, filename='gen'):
    z = np.random.multivariate_normal(mu_avg, cov_avg)
    new_train_melody1 = model.decoder(torch.from_numpy(z).view(1,1,-1).float())
    new_train_melody2 = safe_map(rmap, new_train_melody1.view(-1).detach().numpy())
    const_offset = np.arange(len(new_train_melody2))

    melody_train = chords_n_notes(new_train_melody2, const_offset, 1)
    melody_train = m21.stream.Stream(melody_train) 
    melody_train.write('midi', os.path.join('./music/', filename+'.mid'))
    return new_train_melody1