import numpy as np

# ISO 226 40 Phon Equal-Loudness-Level Contour
iso226_equal_loudness_level_dict = {
    20 : 99.85,
    25 : 93.94,
    31.5 : 88.17,
    40 : 82.63,
    50 : 77.78,
    63 : 73.08,
    80 : 68.48,
    100 : 64.37,
    125 : 60.59,
    160 : 56.70,
    200 : 53.41,
    250 : 50.40,
    315 : 47.58,
    400 : 44.98,
    500 : 43.05,
    630 : 41.34,
    800 : 40.06,
    1000 : 40.01,
    1250 : 41.82,
    1600 : 42.51,
    2000 : 39.23,
    2500 : 36.51,
    3150 : 35.61,
    4000 : 36.65,
    5000 : 40.01,
    6300 : 45.83,
    8000 : 51.80,
    10000 : 54.28,
    12500 : 51.49,
}

class fft_noise_gate():
    def __init__(self, sample_rate, fft_size, attack_ms, release_ms, avr_attack_ms, avr_release_ms, threshhold, ratio, min_gain, smoosing_factor, auto_ratio, avr_auto_ratio):
        self.attack_ms = attack_ms
        self.release_ms = release_ms
        self.avr_attack_ms = avr_attack_ms
        self.avr_release_ms = avr_release_ms        
        self.threshhold = threshhold
        self.ratio = ratio
        size = fft_size // 2 + 1
        self.bin = sample_rate / fft_size
        self.avr_gain_bin = np.zeros(size)
        self.gain = np.zeros(size)
        self.bin_gain_smoothed = np.zeros(size)
        self.bin_noise_lvl = np.zeros(size)
        for i in range(size):
            self.avr_gain_bin[i] = 1
            self.gain[i] = 1
            self.bin_gain_smoothed[i] = 1
            self.bin_noise_lvl[i] = -80
        self.attack_step = (100 * fft_size / sample_rate) / self.attack_ms
        self.release_step = (100 * fft_size / sample_rate) / self.release_ms
        self.avr_attack_step = (100 * fft_size / sample_rate) / self.avr_attack_ms
        self.avr_release_step = (100 * fft_size / sample_rate) / self.avr_release_ms        
        self.threshhold = threshhold
        self.ratio = ratio
        self.auto_ratio = auto_ratio
        self.min_gain = min_gain
        self.smoosing_factor = smoosing_factor
        self.timer = 0
        # self.avr_noise_lvl = -80
        # self.avr_auto_ratio = avr_auto_ratio    
        # self.avr_gain = 1  
        # self.avr_gain_smoothed = 1  


    def filter(self, in_buf):
        in_buf[0] = in_buf[0] * 0.001
        in_buf[1] = in_buf[1] * 0.2
        # in_buf[2] = in_buf[2] * 0.5
        # in_buf[3] = in_buf[3] * 0.5        
        return in_buf
       
    def process_bins(self, in_buf):   
        for i in range(len(in_buf)):
            sample = np.abs(in_buf[i])
            sample_log = 20 * np.log10(sample)
            self.bin_noise_lvl[i] = (self.bin_noise_lvl[i] * (self.smoosing_factor - 1) + sample_log) / self.smoosing_factor
            bin_delta = (sample_log - self.bin_noise_lvl[i] / self.auto_ratio) * self.ratio
            bin_gain = np.power(10, bin_delta / 20)
            bin_gain_delta = self.gain[i] - bin_gain
            if bin_gain_delta > self.release_step:
                bin_gain_delta = self.release_step
            if bin_gain_delta < -self.attack_step:
                bin_gain_delta = -self.attack_step  
            self.gain[i] = self.gain[i] - bin_gain_delta
            if (self.gain[i] > 1):
                self.gain[i] = 1
            if (self.gain[i] < self.min_gain):
                self.gain[i] = self.min_gain
            in_buf[i] = in_buf[i] * self.gain[i]
        return in_buf
    
    def process_bins_alt(self, in_buf):  
        avr_noise_lvl = 0
        for i in range(len(in_buf)):
            sample = np.abs(in_buf[i])
            sample_log = 20 * np.log10(sample)
            avr_noise_lvl += sample_log
            self.bin_noise_lvl[i] = (self.bin_noise_lvl[i] * (self.smoosing_factor - 1) + sample_log) / self.smoosing_factor
            bin_delta = (sample_log - self.bin_noise_lvl[i] / self.auto_ratio) * self.ratio
            bin_gain = np.power(10, bin_delta / 20)
            bin_gain_delta = self.gain[i] - bin_gain
            if bin_gain_delta > self.release_step:
                bin_gain_delta = self.release_step
            if bin_gain_delta < -self.attack_step:
                bin_gain_delta = -self.attack_step  
            self.gain[i] = self.gain[i] - bin_gain_delta
            if (self.gain[i] > 1):
                self.gain[i] = 1
            if (self.gain[i] < self.min_gain):
                self.gain[i] = self.min_gain
            in_buf[i] = in_buf[i] * self.gain[i]
        avr_noise_lvl /= len(in_buf)
        return in_buf   
    
    # def process_vad(self, in_buf):   
    #     avr_noise_lvl = 0
    #     avr_gain = 0
    #     for i in range(len(in_buf)):
    #         sample = np.abs(in_buf[i])
    #         sample_log = 20 * np.log10(sample)
    #         avr_noise_lvl = avr_noise_lvl + sample_log
    #         avr_delta = (sample_log - self.avr_noise_lvl / self.avr_auto_ratio) * self.ratio
    #         avr_gain = np.power(10, avr_delta / 20)
    #         avr_gain_delta = self.gain[i] - avr_gain
    #         if avr_gain_delta > self.avr_release_step:
    #             avr_gain_delta = self.avr_release_step
    #         if avr_gain_delta < -self.avr_attack_step:
    #             avr_gain_delta = -self.avr_attack_step  
    #         self.avr_gain_bin[i] = self.avr_gain_bin[i] - avr_gain_delta
    #         if (self.avr_gain_bin[i] > 1):
    #             self.avr_gain_bin[i] = 1
    #         if (self.avr_gain_bin[i] < self.min_gain):
    #             self.avr_gain_bin[i] = self.min_gain
    #         avr_gain += self.avr_gain_bin[i]
    #     self.avr_gain = avr_gain / len(in_buf)
    #     self.avr_noise_lvl = (self.avr_noise_lvl * (self.smoosing_factor - 1) + avr_noise_lvl / len(in_buf)) / self.smoosing_factor
    #     if(self.avr_gain < 0.02):
    #         in_buf[0] = 1.0
    #     return in_buf

    def process(self, in_buf):
        self.filter(in_buf)  
        self.process_bins(in_buf)
        # self.process_vad(in_buf)
        return  in_buf