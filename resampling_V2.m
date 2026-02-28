clear
clc
close all

ts = 5; %sampling time [min]
flag.save = 1;     % = 1 save; = 0 no save;

for pat= [1:100]
    

    patient = ['adult#' sprintf('%03d', pat)]; % This should give you 'adult#098'
    % directories.data = 'C:\Users\teora\Desktop\ECC25\datasetIacono\dt17';
    % directories.data = 'C:\Users\teora\Desktop\ECC25\datasetIacono\dt6_IP';
    % directories.data = 'C:\Users\teora\Desktop\ECC25\datasetJack\Training_data\train_30days_IP_smoothed';
    % directories.data = 'C:\Users\teora\Desktop\ECC25\datasetJack\Testing_data\ts-dt1';
    directories.data = 'C:\Users\LEOPARD\Documents\TesiMagistrale\Machine_Learning\online-learning-based-blood-glucose-prediction-main\datasets-20251003T101856Z-1-001\datasets\Testing_data\ts-dtM_IP';
    % directories.data = 'C:\Users\teora\Desktop\ECC25\datasetJack\Testing_data\ts-dtM_IP';
    directories.patient = [directories.data '\s#' patient '.mat'];

    %directories.patient_G_IR = [directories.data '\G_IR_' num2str(pat)];

    %load data
    load([directories.patient]);
    %load([directories.patient_G_IR]);

    %resampling from 1min to 5min/from 30 sec to 5 min
    G.signals.values(1) = []; G.time(1) = []; G_5 = G.signals.values([1:ts:end]);      %G
    CGM.signals.values(1) = []; CGM.time(1) = []; CGM_5 = CGM.signals.values([1:ts:end]); %CGM
    CGM_IP.signals.values(1) = []; CGM_IP.time(1) = []; CGM_IP_5 = CGM_IP.signals.values([1:ts:end]); %CGM_IP

    carb_intake.signals.values(1) = []; carb_intake.signals.values = carb_intake.signals.values(1:2:end);
    carb_intake.time(1) = []; carb_intake.time = carb_intake.time(1:2:end);
    meal_5 = carb_intake.signals.values(1:ts:end); %meal
    ins_5 = injection.signals.values(2:ts:end)/ts; %insulin
    if pat == 79 && strcmp(directories.data, 'C:\Users\LEOPARD\Documents\TesiMagistrale\Machine_Learning\online-learning-based-blood-glucose-prediction-main\datasets-20251003T101856Z-1-001\datasets\Testing_data\ts-dtM_IP')
        idx_bolus = scenario.meal_announce.time(4:4:end)/ts + 1;
        bolus_5 = ins_5*0;
        bolus_5(idx_bolus) = ins_5(idx_bolus);
        basal_5 = ins_5-bolus_5;
    else
        injection.signals.values(1) = []; 
        injection.time(1) = [];
        idxValidVal=find(basalBolusMem.basal_reconstructed_time==0,1,'last');
        basal_5 = basalBolusMem.basal_reconstructed*100; basal_5(1:idxValidVal+1) = []; basal_5(end+1) = 0; %basal [pmol/min]
        bolus_5 = basalBolusMem.bolus_reconstructed*6000/ts; bolus_5(1:idxValidVal+1) = []; bolus_5(end+1) = 0; %bolus [pmol/min]
    end

    %G_IR_98(1) = [];
    %G_IR_98_5 = G_IR_98(1:ts:end);
    % === Costruzione tabella per CSV ===
    BloodGlucose = CGM_IP_5(:);     % o CGM_5 se preferisci CGM_IP_5(:);
    basal        = basal_5(:);
    bolus        = bolus_5(:);
    CHO          = meal_5(:);

    % Crea tabella
    T = table(CHO, basal, bolus, BloodGlucose);

    % Percorso di salvataggio (Python si aspetta data/site_<id>/dataset.csv)
    site_id = pat; % oppure qualunque ID numerico tu voglia
    save_dir = fullfile('C:\Users\LEOPARD\Documents\TesiMagistrale\Machine_Learning\online-learning-based-blood-glucose-prediction-main\datasets-20251003T101856Z-1-001\datasets\Testing_data\csv-ts-dtM_IP', ['site_' num2str(site_id)]);
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end

    % Salva come CSV
    writetable(T, fullfile(save_dir, 'dataset.csv'));
    disp(['✅ Salvato: ' fullfile(save_dir, 'dataset.csv')]);


    % if flag.save == 1
    %     cd(directories.data)
    %     save(['s#' patient '.mat'],'G','CGM','injection','carb_intake','CGM_5','basal_5','bolus_5','ins_5','meal_5','-append')
    % end


end



%%% ATTENZIONE PAZIENTE 79 %%% DATI CORROTTI 
% carb_intake = data_corretto.carb_intake;
% idx_bolus = data_corretto.scenario.meal_announce.time(4:4:end)/5 + 1;
% bolus_5 = ins_5*0;
% bolus_5(idx_bolus) = ins_5(idx_bolus);
% basal_5 = ins_5-bolus_5;
% BloodGlucose = CGM_5(:);     % o G_5 se preferisci
% basal        = basal_5(:);
% bolus        = bolus_5(:);
% CHO          = meal_5(:);
% T = table(CHO, basal, bolus, BloodGlucose);
% site_id = pat; % oppure qualunque ID numerico tu voglia
% save_dir = fullfile('C:\Users\teora\Desktop\ECC25\online-learning-based-blood-glucose-prediction-main\data\iacono\alarm\G\', ['site_' num2str(site_id)]);
% if ~exist(save_dir, 'dir')
% mkdir(save_dir);
% end
% % Salva come CSV
% writetable(T, fullfile(save_dir, 'dataset.csv'));
% disp(['✅ Salvato: ' fullfile(save_dir, 'dataset.csv')]);