clc;
clear;
close all;

% ==============================
% CONFIGURATION
% ==============================
fs = 173.61;                 % Sampling frequency
basePath = fullfile('MiniProject/Base_Dataset');

folders = {'S','F','N','Z','O'};
labels  = {'ictal','nonictal','nonictal','healthy','healthy'};

featureMatrix = [];
classLabels   = [];

% ==============================f
% VERIFY DATASET EXISTS
% ==============================
if ~isfolder(basePath)
    error('Dataset folder not found: %s', basePath);
end

% ==============================
% FILTERS
% ==============================
[b_bp,a_bp] = butter(4,[0.5 40]/(fs/2),'bandpass');
[b_delta,a_delta] = butter(4,[0.5 4]/(fs/2),'bandpass');
[b_theta,a_theta] = butter(4,[4 8]/(fs/2),'bandpass');
[b_alpha,a_alpha] = butter(4,[8 13]/(fs/2),'bandpass');
[b_beta,a_beta]   = butter(4,[13 30]/(fs/2),'bandpass');
[b_gamma,a_gamma] = butter(4,[30 40]/(fs/2),'bandpass');

% ==============================
% MAIN LOOP
% ==============================
for f = 1:length(folders)

    folderPath = fullfile(basePath, folders{f});
    if ~isfolder(folderPath)
        warning('Missing folder: %s', folderPath);
        continue;
    end

    files = dir(fullfile(folderPath,'*.txt'));
    if isempty(files)
        warning('No .txt files in %s', folderPath);
        continue;
    end

    for k = 1:length(files)

        % LOAD EEG
        eeg = load(fullfile(folderPath,files(k).name));
        eeg = eeg(:)';

        % PREPROCESS
        eeg = eeg - mean(eeg);
        eeg_filt = filtfilt(b_bp,a_bp,eeg);

        % SUBBANDS
        delta = filtfilt(b_delta,a_delta,eeg_filt);
        theta = filtfilt(b_theta,a_theta,eeg_filt);
        alpha = filtfilt(b_alpha,a_alpha,eeg_filt);
        beta  = filtfilt(b_beta,a_beta,eeg_filt);
        gamma = filtfilt(b_gamma,a_gamma,eeg_filt);

        % ==========================
        % FEATURE EXTRACTION
        % ==========================

        % (A) Statistical features (5)
        stat_feats = [
            mean(eeg_filt)
            std(eeg_filt)
            var(eeg_filt)
            skewness(eeg_filt)
            kurtosis(eeg_filt)
        ];

        % (B) Band power features (5)
        band_feats = [
            bandpower(delta,fs,[0.5 4])
            bandpower(theta,fs,[4 8])
            bandpower(alpha,fs,[8 13])
            bandpower(beta, fs,[13 30])
            bandpower(gamma,fs,[30 40])
        ];

        % (C) Entropy features (5)
        shannon_ent = wentropy(eeg_filt,'shannon');
        log_energy  = wentropy(eeg_filt,'log energy');

        approx_ent = approximateEntropy(eeg_filt, ...
            'Dimension', 2, ...
            'Radius', 0.2*std(eeg_filt));

        sample_ent = sampen(eeg_filt,2,0.2*std(eeg_filt));

        psd = pwelch(eeg_filt,[],[],[],fs);
        psd = psd / sum(psd);
        spectral_ent = -sum(psd .* log2(psd + eps));

        entropy_feats = [
            shannon_ent
            log_energy
            approx_ent
            sample_ent
            spectral_ent
        ];

        % FEATURE VECTOR (15)
        featureVector = [stat_feats; band_feats; entropy_feats]';

        if length(featureVector) ~= 15
            error('Feature length mismatch: %d', length(featureVector));
        end

        featureMatrix = [featureMatrix; featureVector];
        classLabels   = [classLabels; string(labels{f})];

    end
end

% ==============================
% FINAL TABLE
% ==============================
if isempty(featureMatrix)
    error('No EEG files processed.');
end

featureNames = {
    'Mean','Std','Variance','Skewness','Kurtosis', ...
    'DeltaPower','ThetaPower','AlphaPower','BetaPower','GammaPower', ...
    'ShannonEntropy','LogEnergyEntropy','ApproxEntropy','SampleEntropy','SpectralEntropy'
};

T = array2table(featureMatrix,'VariableNames',featureNames);
T.Class = classLabels;

writetable(T,'EEG_Feature_Dataset.csv');

disp(['📊 Total samples: ', num2str(size(T,1))]);

% =====================================================
% LOCAL FUNCTION: FAST SAMPLE ENTROPY
% =====================================================
function se = sampen(x,m,r)
    x = x(:);                % column vector
    N = length(x);

    if N <= m+1
        se = NaN;
        return;
    end

    % Create m-length vectors
    Xm = zeros(N-m+1, m);
    for i = 1:m
        Xm(:,i) = x(i:N-m+i);
    end

    % Chebyshev distance matrix
    D = pdist2(Xm, Xm, 'chebychev');
    B = sum(D < r, 2) - 1;       % exclude self-match
    B_total = sum(B);

    % m+1 vectors
    Xm1 = zeros(N-m, m+1);
    for i = 1:m+1
        Xm1(:,i) = x(i:N-m-1+i);
    end
    D1 = pdist2(Xm1, Xm1, 'chebychev');
    A = sum(D1 < r, 2) - 1;
    A_total = sum(A);

    if A_total == 0 || B_total == 0
        se = Inf;
    else
        se = -log(A_total / B_total);
    end
end