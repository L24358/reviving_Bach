import os
import torch
import numpy as np
import music21 as m21
from collections import Counter

'''
### ========== Basic functions ========== ###
'''

def get_notes(file):
    '''
    file: string, midi file name
    -------
    Returns:
        - list, containing all notes
        - dic, key: instrument, value: notes corresponding to instrument
    '''
    songs = m21.instrument.partitionByInstrument(file)
    
    dic = {}; all_notes = []
    for part in songs.parts:
        
        notes = []
        for element in part.recurse():
            
            if isinstance(element, m21.note.Note):
                notes.append([str(element.pitch), float(element.offset)]) 
                all_notes.append(str(element.pitch))
            elif isinstance(element, m21.chord.Chord):
                notes.append([str(element.pitches[0]), float(element.offset)]) 
                all_notes.append(str(element.pitches[0]))
                
        dic[part.id] = notes
    return list(set(all_notes)), dic

def remove_short(instru2corpus, thre=19):
    '''
    Remove songs that are shorter than threshold=thre.
    '''
    for instrument in instru2corpus.keys():
        for song in instru2corpus[instrument]:
            if len(song) <= thre:
                instru2corpus[instrument].remove(song)
    return instru2corpus

def batchify(X):
    '''
    Reshapes all data into the format (batch, features, length).
    '''
    return [torch.from_numpy(np.array(song).reshape(1,1,-1)).float() for song in X]

def breakeven(song_list, song_len, stride):
    '''
    Breaks songs into parts of equal length (song_len), with stride=stride.
    '''
    new_songlist = []
    for song in song_list:
        t = 0; T = len(song)
        
        while t+song_len < T:
            new_songlist.append(song[t:t+song_len])
            t += stride
    return new_songlist

def difference(song):
    '''
    Takes the difference of the offset to use as duration.
    '''
    res = []
    for t in range(len(song)-1):
        res.append([song[t][0], round(song[t+1][1]-song[t][1],2)])
    res.append([song[-1][0], 1])
    return res

def center(song):
    '''
    Adjust song offset so that it starts at 0.
    '''
    res = []
    t_min = song[0][1]
    for t in range(len(song)):
        res.append([song[t][0], song[t][1]-t_min])
    return res

def upsample(song, base=12):
    '''
    Upsample song by its duration.
    '''
    song = difference(song)
    
    res = []
    for note, diff in song:
        rp = int(diff*base)
        res += [[note, 0] for i in range(rp)]
    return res

def nearest(item, l):
    diff = np.absolute(np.array(list(l))-item)
    return list(l)[list(diff).index(min(diff))]

'''
### ========== Utility functions ========== ###
'''

def zip_(X_melody, X_offset, rmap, rmap2, fmap_j):
    N_song = len(X_melody)

    X_joint = []
    for s in range(N_song):
        notes = X_melody[s].squeeze().numpy()
        offs = X_offset[s].squeeze().numpy()
        joint = list(zip([rmap[n] for n in notes], [rmap2[o] for o in offs]))
        song = np.array([fmap_j[jt] for jt in joint])/20.0 # rescale by 20
        X_joint.append(torch.from_numpy(song).float().reshape(1,1,-1))
    return X_joint

def upsample_batch(instru2corpus):
    '''
    Calls upsample() in batches.
    '''
    new_instru2corpus = {}
    for instrument in instru2corpus.keys():
        new_instru2corpus[instrument] = []
        song_list = list(filter(([]).__ne__, instru2corpus[instrument]))
        
        for song in song_list:
            new_instru2corpus[instrument].append(upsample(song))
            
    return new_instru2corpus

def get_notes_batch(files):
    '''
    Calls get_notes() in batches.
    ----------
    Returns:
        - list, containing all notes
        - dic, key: instrument, value: notes corresponding to instrument
    '''
    all_notes, all_dic = [], {}
    for file in files:
        notes, dic = get_notes(file)
        for instrument in dic.keys():
            if instrument not in all_dic.keys(): all_dic[instrument] = []
            all_dic[instrument].append(dic[instrument])
        all_notes += notes
    return all_notes, all_dic

def get_map(Corpus):
    '''
    Corpus: list, contains all notes
    -------
    Returns:
        - list, contains all notes
        - dic, key: notes, value: int
        - dic, key: int, value: notes
    '''
    order = {'C':1,'D':2,'E':3,'F':4,'G':5,'A':6,'B':7, '#':0.5, '-':-0.5}
    
    dic, fmap, rmap = {}, {}, {}
    for elem in list(set(Corpus)):
        if len(elem) == 2: # not black keys
            note, octave = elem; pitch = order[note]
        else:
            note, black, octave = elem; pitch = order[note] + order[black]
        dic[(int(octave), pitch)] = elem
            
    for idx, key in enumerate(sorted(dic)):
        fmap[dic[key]] = idx  
        rmap[idx] = dic[key]
        
    return Corpus, fmap, rmap

def get_midis(filepath):
    '''
    filepath : string, path that contains midi files
    -------
    Returns:
        - list, containing music21-processed scores
        - string, filenames corresponding to the scores in all_midis
    '''
    all_midis, fnames = [], []
    for f in os.listdir(filepath):
        if f.endswith(".mid"):
            fnames.append(f)
            tr = os.path.join(filepath, f)
            midi = m21.converter.parse(tr)
            all_midis.append(midi)
    return all_midis, fnames

def train_test_split(instru2corpus, instrument, fmap, song_len, stride,
                     train_percentage=0.8, seed=None, process=None):
    '''
    Splits data into training and test (validation) set.
    '''
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

def rmap_safe(rmap, melody):
    '''
    rmap: dic, key: int, value: note
    melody: array-like, containing notes
    -------
    Returns: np.array, contains notes
    '''
    maxnum = max(rmap.keys())
    
    notes = []
    for num in melody:
        if (num <= maxnum) and (num > 0): notes.append(rmap[int(num)])
        elif num > maxnum: notes.append(rmap[maxnum])
        else: notes.append(rmap[0])
    return np.array(notes)

def fmap_offset(X, fmap, song_len):
    '''
    Maps offset to an int.
    '''
    new_X = []
    for song in X:
        shape = song.shape
        
        offs = []
        for off in song.view(-1):
            off_near = fmap[nearest(off.item(), fmap.keys())]
            offs.append(off_near)
        offs = torch.from_numpy(np.array(offs)).view(*shape)
        new_X.append(offs)
    new_X = [song.float() for song in new_X]
    return new_X

def get_map_offset(X):
    X_np = np.array([item.numpy() for item in X])
    Corpus = sorted(list(set(X_np.reshape(-1))))
    fmap, rmap, count = {}, {}, 0
    for off in Corpus:
        fmap[round(float(off),2)] = count
        rmap[count] = off
        count += 1
    return Corpus, fmap, rmap

def get_map_offset_v2(instru2corpus, instrument):
    Corpus = []
    for num in range(len(instru2corpus[instrument])):
        song = np.array([float(off[1]) for off in np.array(instru2corpus[instrument][num])])
        song = song[1:]-song[:-1]
        Corpus += list([round(off,2) for off in song])+[1]
    Corpus = list(set(Corpus))
    
    fmap, rmap = {}, {}
    for idx, key in enumerate(sorted(Corpus)):
        fmap[key] = idx
        rmap[idx] = key
    return Corpus, fmap, rmap

def get_joint_map(fmap, fmap2):
    count = 0
    
    fmap_j, rmap_j = {}, {}
    for note in fmap.keys():
        for off in fmap2.keys():
            fmap_j[(note, off)] = count
            rmap_j[count] = [note, off]
            count += 1
    return fmap_j, rmap_j

def gen_stream(snippet, offset,
                   base=1, output=False, fname=''):
    '''
    snippet: array-like, containing notes
    offset: array-like, containing offset of notes
    -------
    Returns: m21.stream.Stream object
    Outputs: midi file
    '''
    
    # construct Stream object
    Melody = []
    for s in range(len(snippet)):
        note_snip = m21.note.Note(snippet[s])
        note_snip.offset = offset[s]
        Melody.append(note_snip)
    Melody_midi = m21.stream.Stream(Melody)
    
    # output as midi file
    if output:
        Melody_midi.write('midi', os.path.join('./music/', fname+'.mid'))
    return Melody_midi

def gen_reconstruction(model1, model2, X_train_melody, X_train_offset, rmap,
                       base=1, idx=None):
    '''
    Generates reconstructed and original midi files.
    '''
    # set song index
    if idx == None: idx = np.random.randint(0, len(X_train_melody))
    
    recons_train_melody = model1(X_train_melody[idx])[0]
    recons_train_melody = rmap_safe(rmap, recons_train_melody.view(-1).detach().numpy())
    ori_train_melody = rmap_safe(rmap, X_train_melody[idx].view(-1).numpy())
    
    const_offset = np.arange(len(recons_train_melody))
    ori_train_offset = X_train_offset[idx].view(-1).numpy()
    if model2 != None:
        recons_train_offset = model2(X_train_offset[idx])[0].view(-1).detach().numpy()
        gen_stream(recons_train_melody, recons_train_offset, base=base, output=True,
                   fname='recon(f)_'+str(idx)) # full reconstruction
        
    gen_stream(recons_train_melody, ori_train_offset, base=base, output=True,
               fname='recon(d)_'+str(idx)) # cheat using original duration
    
    gen_stream(ori_train_melody, ori_train_offset, base=base, output=True,
               fname='original(f)_'+str(idx))
    
    gen_stream(recons_train_melody, const_offset, base=base, output=True,
               fname='recon(c)_'+str(idx)) # constant duration
    
    gen_stream(ori_train_melody, const_offset, base=base, output=True,
              fname='original(c)_'+str(idx))

def gen_generation_vae(mu_avg, cov_avg, model, rmap, base=1, fname='new'):
    '''
    Generates new midi files (for VAE only).
    '''
    z = np.random.multivariate_normal(mu_avg, cov_avg)
    
    new_train_melody1 = model.decoder(torch.from_numpy(z).view(1,1,-1).float())
    new_train_melody2 = rmap_safe(rmap, new_train_melody1.view(-1).detach().numpy())
    const_offset = np.arange(len(new_train_melody2))

    gen_stream(new_train_melody2, const_offset, base=base, output=True,
               fname=fname)
    return new_train_melody1

def gen_generation_ae(mu_avg, cov_avg, model, rmap, base=1, fname='new'):
    '''
    Generates new midi files (for AE only).
    '''
    z = np.random.multivariate_normal(mu_avg, cov_avg)
    
    new_train_melody1 = model.decoder(torch.from_numpy(z).view(1,4,-1).float())
    new_train_melody2 = rmap_safe(rmap, new_train_melody1.view(-1).detach().numpy())
    const_offset = np.arange(len(new_train_melody2))
    
    gen_stream(new_train_melody2, const_offset, base=base, output=True,
               fname=fname)
    return new_train_melody1