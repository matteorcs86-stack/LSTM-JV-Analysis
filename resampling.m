clear
clc
close all

ts = 5; %sampling time [min]
flag.save = 1;     % = 1 save; = 0 no save;

for pat= [38]

    patient = subjectstr(pat, 'adult'); % This should give you 'adult#098'
    directories.data = 'D:\ucl eee\project_data_more';
    directories.patient = [directories.data '\s#' patient '.mat'];

    %directories.patient_G_IR = [directories.data '\G_IR_' num2str(pat)];
 
    %load data
    load([directories.patient]);
    %load([directories.patient_G_IR]);

    %resampling from 1min to 5min/from 30 sec to 5 min
    G.signals.values(1) = []; G.time(1) = []; G_5 = G.signals.values([1:ts:end]);      %G
    CGM.signals.values(1) = []; CGM.time(1) = []; CGM_5 = CGM.signals.values([1:ts:end]); %CGM
    
    carb_intake.signals.values(1) = []; carb_intake.signals.values = carb_intake.signals.values(1:2:end); 
    carb_intake.time(1) = []; carb_intake.time = carb_intake.time(1:2:end);
    meal_5 = carb_intake.signals.values(1:ts:end); %meal

    injection.signals.values(1) = []; ins_5 = injection.signals.values(1:ts:end)/ts; %insulin
    injection.time(1) = [];
    basal_5 = basalBolusMem.basal_reconstructed*100; basal_5(1:576) = []; basal_5(end+1) = 0; %basal [pmol/min]
    basalBolusMem.basal_reconstructed_time(1:576) = [];
    bolus_5 = basalBolusMem.bolus_reconstructed*6000/ts; bolus_5(1:576) = []; bolus_5(end+1) = 0; %bolus [pmol/min]
    basalBolusMem.bolus_reconstructed_time(1:576) = [];

    %G_IR_98(1) = [];
    %G_IR_98_5 = G_IR_98(1:ts:end);

    if flag.save == 1
        cd(directories.data)
        save(['s#' patient '.mat'],'G','CGM','injection','carb_intake','CGM_5','basal_5','bolus_5','ins_5','meal_5','-append')
    end


end