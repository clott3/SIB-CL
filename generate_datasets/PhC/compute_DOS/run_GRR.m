function run_GRR(sgnum, h5prefix, nsam, rootdir)
    % function DOS_GGR
    %% DOS calculation using generalized GR (GGR) method
    % The program is for DOS calculation using GGR method, referring
    % to article "Generalized Gilat-Raubenheimer Method for Density-of-States 
    % Calculation in Photonic Crystals". For more information, please refer to our website:
    % https://github.com/boyuanliuoptics/DOS-calculation/edit/master/DOS_GGR.m
    % The first edition is finished in Nov. 20th, 2017.
    %% Important notice for initial parameters!!!
    % Necessary parameters: 
    % 0. three files include band frequencies on high symmetry points,
    % band frequencies in the whole Brillouin zone, group velocities in the
    % whole Brillouin zone; 
    % 1. the reciprocal vectors; 2. the number of k points; 3. number of bands.
    % Optional parameters: 4. maximum and minimum of band frequency (w_max, w_min); 
    % 5. resolution about the frequency  (N_w); 6. inter number of points between 
    % two high symmetry k points; 7. parameters about plot like color, fontsize, etc;
    nsam
    % Settings for DOS
    ceilDOS = 1;   % impose ceiling on DOS array (1) or not (0) 
    maxDOS=40;   
    w_max_custom = -1; % the range of frequency, '-1' denotes default settings
    w_min_custom=0;

    % Input parameters here 
%     nsam = 6035;
    kin = 25;
    N_band = 10;
    inputdir = strcat(rootdir,'txt_',string(h5prefix),'_sg',string(sgnum))
    outputdir = strcat(rootdir,'DOS_',string(h5prefix),'_sg',string(sgnum))
    
    if ~exist(outputdir, 'dir')
      mkdir(outputdir);
    end

    for sample=1:nsam 

        if strlength(string(sample))==1
            idx = strcat('_','0',string(sample));
        else
            idx = strcat('_',string(sample));
        end

        file_bandline=strcat(inputdir,'/outband',idx,'.txt');

        file_bandmap = strcat(inputdir,'/trifreqGRR',idx,'.txt');
        file_velocity= strcat(inputdir,'/trivelGRR',idx,'.txt');

        file_DOSdata=strcat(outputdir,'/DOS_GRR',idx,'.txt');    % save file for Density of states data

        %% reciprocal vectors for square lattice
        reciprocalvector1=[1 0 0];
        reciprocalvector2=[0 1 0];
        reciprocalvector3=[0 0 0];

        num_kpoints=[kin,kin,1];

        N_w_custom=20000;       % denotes the resolution of frequency : dw = (w_max - w_min) / N_w

        %% Initialization and import data
        % the reciprocal vectors initialization
        vectorb1=reciprocalvector1;
        vectorb2=reciprocalvector2;
        vectorb3=reciprocalvector3;
        vectorsb=[vectorb1;vectorb2;vectorb3];

        % Nx,Ny,Nz is the number of k points along the x,y,z axis. N_kpoints is the
        % total number of k points in Brillouin zone
        n_kpoints=prod(num_kpoints);
        N_kpoints=N_band*n_kpoints;

        % import data
        % the two importing txt files are arranged as matrix of N*1 and N*3
        dataw=importdata(file_bandmap);
        datav_original=importdata(file_velocity);   % the real group velocities
        datav=(vectorsb*datav_original')';     % the transformed group velocities

        if w_max_custom==-1
            w_max=1.05*max(dataw); % the maximum of frequency should be larger than max(dataw) a little
        else
            w_max=w_max_custom;
        end

        if w_min_custom==-1
            w_min=0;
        else
            w_min=w_min_custom;
        end

        itmd_v=sort(abs(datav),2,'descend');   % intermediate velocity: v1 >= v2 >= v3

        % N_w=20*N_kpoints;       % divide the frequency region into N_w part
        N_w=N_w_custom;
        step_w=(w_max-w_min)/N_w;      % the resolution of frequency
        hside=1/num_kpoints(2)/2;    % half of side length of one transfromed cube
        DOSarray=zeros(N_w+1,1);       % initialze the density of states array

        w1=hside*abs(itmd_v(:,1)-itmd_v(:,2)-itmd_v(:,3));
        w2=hside*(itmd_v(:,1)-itmd_v(:,2)+itmd_v(:,3));
        w3=hside*(itmd_v(:,1)+itmd_v(:,2)-itmd_v(:,3));
        w4=hside*(itmd_v(:,1)+itmd_v(:,2)+itmd_v(:,3));

        %% DOS calculation
        % principle of calculation process can be found in our article
        % "Generalized Gilat-Raubenheimer Method for Density-of-States Calculation
        % in Photonic Crystals"
        for num_k=1:N_kpoints
            n_w_kcenter=round((dataw(num_k)-w_min)/step_w);
            v=norm(datav(num_k,:));
            v1=itmd_v(num_k,1);
            v2=itmd_v(num_k,2);
            v3=itmd_v(num_k,3);

            flag_delta_n_w=0;       % first time compute delta_n_w = 1
            for vdirection=0:1      % two velocity directions denote w-w_k0 > 0 and <0
                for delta_n_w=1:N_w
                    n_tmpt=n_w_kcenter+(-1)^vdirection*(delta_n_w-1);
                    delta_w=abs(dataw(num_k)-(n_tmpt*step_w+w_min));
                    if delta_w<=w1(num_k)
                        if v1>=v2+v3
                            DOScontribution=4*hside^2/v1;
                        else
                            DOScontribution=(2*hside^2*(v1*v2+v2*v3+v3*v1)-...
                                (delta_w^2+(hside*v)^2))/v1/v2/v3;
                        end
                    elseif delta_w<w2(num_k)
                        DOScontribution=(hside^2*(v1*v2+3*v2*v3+v3*v1)-...
                            hside*delta_w*(-v1+v2+v3)-(delta_w^2+hside^2*v^2)/2)/v1/v2/v3;
                    elseif delta_w<w3(num_k)
                        DOScontribution=2*(hside^2*(v1+v2)-hside*delta_w)/v1/v2;
                    elseif delta_w<w4(num_k)
                        DOScontribution=(hside*(v1+v2+v3)-delta_w)^2/v1/v2/v3/2;
                    else
                        break;
                    end
                    if DOScontribution>8*hside^3/step_w
                        DOScontribution=8*hside^3/step_w;
                    end

                    if delta_n_w==1       % when delta_n_w == 1, we only compute it once
                        if flag_delta_n_w==0
                            DOSarray(n_tmpt+1)=DOSarray(n_tmpt+1)+DOScontribution;
                            flag_delta_n_w=1;
                        end
                        continue;
                    else
                        if (n_tmpt>=0)&&(n_tmpt<=N_w)
                            DOSarray(n_tmpt+1)=DOSarray(n_tmpt+1)+DOScontribution;
                        end
                    end
                end
            end
        end

        %output DOS data into output.txt
        if num_kpoints(3)==1    % the structure is 2 dimension
            DOSarray=DOSarray*num_kpoints(2);
        end
        if num_kpoints(1)*2==num_kpoints(2)     % the structure has time-reversal symmetry
            DOSarray=DOSarray*2;
        end

        if ceilDOS ==1
            DOSarray(DOSarray>maxDOS)=maxDOS;
        end

        file_output=fopen(file_DOSdata,'wt');

        for nprint_w=1:(N_w+1)
            fprintf(file_output,'%.10f %.10f\n',w_min+step_w*(nprint_w-1),DOSarray(nprint_w));  
        end
        fclose(file_output);

    %     figure;
    %     plot(DOSarray)

    end   
end 