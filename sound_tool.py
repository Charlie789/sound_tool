import librosa
import numpy as np
import scipy.fftpack as fftpk
import soundfile
from librosa import display
from matplotlib import pyplot as plt
from scipy import fft, signal
from scipy.signal import butter, lfilter

loaded_files = {}


def main_menu():
    print()
    print('** MENU **')
    print('1. Pokaż wykresy')
    print('2. Wczytaj plik dźwiękowy')
    print('3. Odwróć plik')
    print('4. Filtr pasmowy')
    print('5. Filtr dolnoprzepustowy')
    print('0. Koniec')

    choice = input('Wybierz: ')
    if choice == '1':
        draw_plot_time(loaded_files)
    elif choice == '2':
        file_name = input('Podaj nazwe pliku: ')
        load_file(file_name)
    elif choice == '3':
        reverse_file(chose_file())
    elif choice == '4':
        band_pass_filter(chose_file(), input('Dolny zakres filtru [Hz]: '), input('Górny zakres filtru [Hz]: '))
    elif choice == '5':
        butter_lowpass_filter(chose_file(), input('Częstotliwość odcięcia [Hz]: '))
    elif choice == '0':
        exit(0)
    else:
        print('Błędny wybór')


def chose_file():
    print('Wybierz plik:')
    file_list = []
    for index, file_name in enumerate(loaded_files):
        file_list.append(file_name)
        print(f'{index}. {file_name}')
    selected_index = int(input())
    return file_list[selected_index]


def ask_load_file(file_name):
    while True:
        chose = input('Czy chcesz dodać nowy plik do listy? [T/n]')
        if len(chose) == 0 or chose.lower() == 't':
            load_file(file_name)
            return
        elif chose.lower() == 'n':
            return
        else:
            print('Błędna odp')


def load_file(file_name):
    try:
        file = librosa.load(file_name, sr=None, mono=True, offset=0, duration=None)
    except FileNotFoundError:
        print(f'Nie znaleziono "{file_name}", spróbuj jeszcze raz')
        return

    print(f'''
    Pomyślnie załadowano: {file_name}
    Liczba próbek: {len(file[0])}
    Częstotliwość próbkowania: {file[1]}
    Czas trwania: {len(file[0]) / file[1]}
    ''')

    loaded_files[file_name] = file


def draw_plot_time(files):
    plt.figure()
    subplt_index = 1
    for file_name in files:
        file = files[file_name]
        # wykres w dziedzinie czasu
        plt.subplot(len(files), 2, subplt_index)
        librosa.display.waveplot(y=file[0], sr=file[1])
        plt.title(file_name)
        plt.ylabel('Amplituda')
        plt.xlabel('Czas [s]')
        subplt_index = subplt_index + 1

        #wykres w dziedzinie częstotliwości
        T = 1 / file[1]
        FFT = abs(fft.fft(file[0]))
        freqs = fftpk.fftfreq(len(FFT), T)
        ax = plt.subplot(len(files), 2, subplt_index)
        plt.plot(freqs[range(len(FFT)//2)], FFT[range(len(FFT)//2)])
        plt.xlabel('Częstotliwość [Hz]')
        ax.set_ylim([0, 100])
        subplt_index = subplt_index + 1

    plt.tight_layout()
    plt.show()


def reverse_file(file_name):
    new_file_name = file_name[:-4] + '_reverse.wav'
    soundfile.write(new_file_name, np.flip(loaded_files[file_name][0]), loaded_files[file_name][1])
    ask_load_file(new_file_name)


def band_pass_filter(file_name, low_cut, high_cut):
    new_file_name = file_name[:-4] + '_band_pass.wav'

    nyq = 0.5 * loaded_files[file_name][1]
    low = float(low_cut) / nyq
    high = float(high_cut) / nyq

    order = 2

    b, a = signal.butter(order, [low, high], 'bandpass', analog=False)
    y = signal.filtfilt(b, a, loaded_files[file_name][0], axis=0)
    soundfile.write(new_file_name, y, loaded_files[file_name][1])
    ask_load_file(new_file_name)


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(file_name, cutoff):
    new_file_name = file_name[:-4] + '_low.wav'

    b, a = butter_lowpass(float(cutoff), loaded_files[file_name][1], order=5)
    y = lfilter(b, a, loaded_files[file_name][0])

    soundfile.write(new_file_name, y, loaded_files[file_name][1])
    ask_load_file(new_file_name)


if __name__ == '__main__':
    while True:
        main_menu()
