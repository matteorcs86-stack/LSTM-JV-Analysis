%% computeMetrics.m (VERSIONE AGGIORNATA - PRED + SUMMARY ROBUSTO)
clc
clear
close all

%% =========================
%  Folders (DEFINITIVO)
%  =========================
folders.main = 'C:\Users\LEOPARD\Documents\TesiMagistrale\Machine_Learning';
folders.code = fullfile(folders.main, 'online-learning-based-blood-glucose-prediction-main');

% Cartella dove sta questo script + funzioni MATLAB di post-processing
folders.elab = fullfile(folders.code, 'Elab_dati');

% Root dataset originali (MATLAB .mat dei pazienti)
folders.dataRoot = fullfile(folders.code, ...
    'datasets-20251003T101856Z-1-001', 'datasets', 'Testing_data');

% Root predizioni prodotte da Python
folders.batchLogs = fullfile(folders.code, 'batch_logs');

% =========================
% SCEGLI QUI SCENARIO & ORIZZONTE
% =========================
scenarioName = 'ts-dtM';  % es: 'ts-dt1', 'ts-dt2', 'dt6', 'ts-dt1_IP', ...
PH = 40;                  % prediction horizon (min): 40 oppure 60

% Dataset originale (true)
folders.testData = fullfile(folders.dataRoot, scenarioName);

% Predizioni (cartella dove stanno predictions_vs_targets_site_XXX.mat)
folders.testPredictions = fullfile( ...
    folders.batchLogs, ...
    sprintf('Horizon_%d', PH), ...
    sprintf('Log_%s-H%d', scenarioName, PH) ...
);

% Output (Results_v2 sotto Elab_dati)
folders.saveRoot = fullfile(folders.elab, 'Results_v2');
folders.save = fullfile(folders.saveRoot, sprintf('%s_H%d', scenarioName, PH));
if ~exist(folders.save,'dir')
    mkdir(folders.save);
end


%% =========================
% Patients
% =========================
patients = 1:100;
%patients = 1; da usare per test di funzionamento/debug
pat_to_plot = 1;  % paziente di esempio per grafico
%pat_to_plot = 1; da usare per test di funzionamento/debug


%% =========================
% Parameter (allineati al tuo script)
% =========================
L = 300;     % lookback window [min]
Ts = 5;      % sampling time [min]
w = L + PH;  % window length [min] (come nel tuo script)

%% Flags
flags.viewGraphs = 0;
flags.doPred = 1;

% Parte Alarm: per ora OFF per evitare errori di path finché non la riallineiamo
flags.doAlarm = 0;


%% =========================
% Prediction results
% =========================
if flags.doPred
    disp('### Prediction Result ###')

    saveFile = fullfile(folders.save, 'pred_indices.mat');

    if ~exist(saveFile,'file')
        % Preallocazione (cell array indicizzata per id paziente)
        y_pred = cell(1, max(patients));
        y_true = cell(1, max(patients));
        pred_indices = cell(1, max(patients));

        for pat = patients
            pat_str = sprintf('s#adult#%03d', pat);

            % --- Dataset originale ---
            testFile = fullfile(folders.testData, [pat_str '.mat']);
            if ~isfile(testFile)
                warning('Dataset mancante per paziente %d: %s (skip)', pat, testFile);
                continue;
            end
            test_data = load(testFile);

            % --- Predizioni rinominate ---
            predFile = fullfile(folders.testPredictions, ...
                sprintf('predictions_vs_targets_site_%03d.mat', pat));

            if ~isfile(predFile)
                warning('Predizioni mancanti per paziente %d: %s (skip)', pat, predFile);
                continue;
            end
            pred_data = load(predFile);

            % --- Predizione (ultima colonna = orizzonte PH) ---
            y_pred{pat} = pred_data.profiles_hat(:, end);

            % --- Target vero (stessa logica del tuo script originale) ---
            % NB: qui assumiamo che G.signals.values sia a campionamento 1 min
            y_true{pat} = test_data.G.signals.values(w+PH : Ts : end-Ts);

            % --- Metriche per paziente ---
            e = y_pred{pat} - y_true{pat};
            denom = norm(y_true{pat} - mean(y_true{pat}), 2);

            pred_indices{pat}.COD  = 100 * (1 - (norm(e,2)^2) / (denom^2));
            pred_indices{pat}.FIT  = 100 * (1 -  norm(e,2)     /  denom);
            pred_indices{pat}.rho  = corr(y_pred{pat}, y_true{pat});
            pred_indices{pat}.RMSE = sqrt(mean(e.^2));

            [pred_indices{pat}.DU, pred_indices{pat}.DD] = calc_delay(y_true{pat}, y_pred{pat});
        end

        % Salva
        save(saveFile, 'pred_indices', 'y_pred', 'y_true');

    else
        % Carica (versione coerente con fullfile)
        load(fullfile(folders.save, 'pred_indices.mat'), 'pred_indices', 'y_pred', 'y_true');
    end

    % --- Pazienti validi ---
    patients_valid = find(~cellfun(@isempty, y_pred));
    patients_missing = setdiff(patients, patients_valid);

    % --- Vettori metriche (uno per paziente) ---
    FIT_all  = arrayfun(@(i) pred_indices{i}.FIT,  patients_valid);
    RMSE_all = arrayfun(@(i) pred_indices{i}.RMSE, patients_valid);
    DU_all   = arrayfun(@(i) pred_indices{i}.DU,   patients_valid);
    DD_all   = arrayfun(@(i) pred_indices{i}.DD,   patients_valid);

    % --- Summary globale (salvato SEMPRE) ---
    summary = struct();
    summary.scenarioName = scenarioName;
    summary.PH = PH;
    summary.L  = L;
    summary.w  = w;
    summary.Ts = Ts;

    summary.patients_requested = patients;
    summary.patients_valid = patients_valid;
    summary.patients_missing = patients_missing;

    summary.N_requested = numel(patients);
    summary.N_valid = numel(patients_valid);
    summary.N_missing = numel(patients_missing);

    summary.FIT  = summarize_metric(FIT_all);
    summary.RMSE = summarize_metric(RMSE_all);
    summary.DU   = summarize_metric(DU_all);
    summary.DD   = summarize_metric(DD_all);

    save(fullfile(folders.save, 'pred_summary.mat'), 'summary');
    disp(['Salvato: ' fullfile(folders.save, 'pred_summary.mat')]);

    % --- Output leggibile a schermo (opzionale ma utile) ---
    print_metric('FIT',  summary.FIT,  summary.N_valid, summary.N_requested);
    print_metric('RMSE', summary.RMSE, summary.N_valid, summary.N_requested);
    print_metric('DU (samples)', summary.DU, summary.N_valid, summary.N_requested);
    print_metric('DD (samples)', summary.DD, summary.N_valid, summary.N_requested);

    if ~isempty(patients_missing)
        disp(['ℹ️ Pazienti mancanti (predizioni non trovate): ' num2str(patients_missing)]);
    end

    % --- Plot pazienti selezionati ---
    plot_pred(pat_to_plot, y_pred, y_true, folders, Ts);
end


%% =========================
% Alarm results (DISATTIVATO)
% =========================
if flags.doAlarm
    % Qui lasciamo la tua logica originale, ma va riallineata ai tuoi path.
    % Quando vorrai riattivarla, dimmi:
    % - dove sono i file true per l'alarm (dataset alarm)
    % - dove sono le predizioni alarm (cartelle / nomi)
    %
    % Per ora OFF per evitare errori in test.
end


%% =========================
% Funzioni locali (helper)
% =========================
function S = summarize_metric(x)
    x = x(:);
    x = x(isfinite(x));  % rimuove NaN/Inf

    S = struct();
    S.N = numel(x);

    if S.N == 0
        S.mean = NaN; S.std = NaN;
        S.median = NaN; S.prc25 = NaN; S.prc75 = NaN; S.iqr = NaN;
        S.lillie_reject = NaN;
        return;
    end

    S.mean   = mean(x);
    S.std    = std(x);
    S.median = median(x);
    S.prc25  = prctile(x,25);
    S.prc75  = prctile(x,75);
    S.iqr    = S.prc75 - S.prc25;

    if S.N >= 4
        S.lillie_reject = lillietest(x); % 0 = non rifiuta normalità
    else
        S.lillie_reject = NaN;
    end
end

function print_metric(name, S, Nvalid, Nreq)
% Stampa una metrica in modo auto-esplicativo:
% - se distribuzione ~ normale: mean ± std
% - altrimenti: median (IQR: p25–p75)
% - se N<4: stampa entrambe (nessun test di normalità)

    if nargin < 3
        Nvalid = S.N; 
        Nreq   = S.N;
    end

    if isempty(S) || ~isfield(S,'N') || S.N == 0
        disp(sprintf('%s (N=%d/%d): N=0', name, Nvalid, Nreq));
        return;
    end

    prefix = sprintf('%s (N=%d/%d): ', name, Nvalid, Nreq);

    % Caso N<4: niente lillietest -> stampa entrambe le sintesi
    if isnan(S.lillie_reject)
        disp(sprintf([prefix ...
            'mean=%.4g ± %.4g | median=%.4g (IQR: %.4g–%.4g)'], ...
            S.mean, S.std, S.median, S.prc25, S.prc75));
        return;
    end

    % Caso normale (non rifiuta normalità)
    if S.lillie_reject == 0
        disp(sprintf([prefix 'mean=%.4g ± %.4g'], S.mean, S.std));
    else
        % Caso non normale (rifiuta normalità)
        disp(sprintf([prefix 'median=%.4g (IQR: %.4g–%.4g)'], ...
            S.median, S.prc25, S.prc75));
    end
end


%% =========================
% Plot: stessa funzione che avevi (con piccole sistemazioni)
% =========================
function plot_pred(pat_to_plot, y_pred, y_true, folders, Ts)
    samples_per_hour = 60/Ts;         % es: 12 campioni/ora se Ts=5
    samples_per_day  = 24*samples_per_hour; % 288
    max_days_to_plot = 2;
    xmax = samples_per_day * max_days_to_plot; % 576

    n = numel(pat_to_plot);
    f5 = figure('Color','w','Position',[100 100 900 max(260, 220*n)]);
    t  = tiledlayout(n,1,'TileSpacing','compact','Padding','compact');

    for i = 1:n
        pat = pat_to_plot(i);

        ax = nexttile;
        if pat > numel(y_true) || isempty(y_true{pat}) || isempty(y_pred{pat})
            title(ax, sprintf('Patient %03d (missing)', pat));
            continue;
        end

        p1 = plot(y_true{pat}, 'LineWidth',2); hold on;
        p2 = plot(y_pred{pat}, 'LineWidth',2, 'LineStyle','--');

        grid on; box on;

        xlim([1 min(xmax, numel(y_true{pat}))]);
        ylim([20 max(y_true{pat})]);

        hours_to_show = 0:6:(24*max_days_to_plot);
        xticks(hours_to_show * samples_per_hour);
        xticklabels(compose('%02d:00', hours_to_show));
        xlabel('Time [h]');

        title(ax, sprintf('Patient %03d', pat));

        if i == 1
            legend([p1 p2], {'True','Pred'}, 'Location','best','Box','off');
        end
    end

    ylabel(t, 'Glucose [mg/dL]');

    exportgraphics(f5, fullfile(folders.save,'pred.png'), 'Resolution',300);
    exportgraphics(f5, fullfile(folders.save,'pred.pdf'));
    savefig(f5, fullfile(folders.save,'pred.fig'));
end