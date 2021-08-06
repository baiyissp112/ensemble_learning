# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:45:53 2021

@author: Administrator
"""
#预测天，整景数据
import os
import pickle
import numpy as np
from osgeo import gdal

def get_filedir(path, keyname, time):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path) for file in files if keyname in file and time in file and 'tif' in file[-3:] ]

def get_tif_path(root_dir):
    
    CO_path = get_filedir(root_dir,"_CO_", time)
    HCHO_path = get_filedir(root_dir,"_HCHO_", time)
    NO2_path = get_filedir(root_dir,"_NO2_", time)
    AOD_path = get_filedir(root_dir,"MCD19A2", time)
    
    alnid_path = get_filedir(root_dir,"_alnid_", time)
    vieo_path = get_filedir(root_dir,"_p77.162_", time)
    vino_path = get_filedir(root_dir,"_p78.162_", time)
    
    sp_path = get_filedir(root_dir,"_sp_", time)
    rh_path = get_filedir(root_dir,"_r_", time)
    blh_path = get_filedir(root_dir,"_blh_", time)
    cbh_path = get_filedir(root_dir,"_cbh_", time)
    ssrd_path = get_filedir(root_dir,"_ssrd_", time)
    t2m_path = get_filedir(root_dir,"_t2m_", time)
    u_path = get_filedir(root_dir,"_u10_", time)
    v_path = get_filedir(root_dir,"_v10_", time)
    
    return CO_path,HCHO_path,NO2_path,vieo_path,vino_path,sp_path,rh_path,blh_path,ssrd_path,t2m_path,u_path,v_path,alnid_path,cbh_path,AOD_path



def get_img(filePath):
    dataset = gdal.Open(filePath)
    
    nXSize = dataset.RasterXSize #列数
    nYSize = dataset.RasterYSize #行数
    adfGeoTransform = dataset.GetGeoTransform() #地理坐标信息
    
    im_data = dataset.ReadAsArray(0,0,nXSize,nYSize)
    im_data = im_data.ravel() # 数组降为一维，默认行优先，传入‘F’表示列优先
    im_data = im_data.reshape((-1, 1))

    if '_r_' in filePath:
        coord_x = [] # 用于存储每个像素的X坐标
        coord_y = [] # 用于存储每个像素的y坐标
        for i in range(nYSize):
            for j in range(nXSize):
                px = adfGeoTransform[0] + j * adfGeoTransform[1] + i * adfGeoTransform[2]
                py = adfGeoTransform[3] + j * adfGeoTransform[4] + i * adfGeoTransform[5]
                coord_x.append(px)
                coord_y.append(py)
        coord_x = np.array(coord_x).reshape((-1, 1))
        coord_y = np.array(coord_y).reshape((-1, 1))
        im_month = int(filePath.split('\\')[-1][9:11])
        month_arr = np.full([len(im_data), 1], im_month)
        im_date = int(filePath.split('\\')[-1][11:13]) # 获得日期
        date_arr = np.full([len(im_data), 1], im_date) # 日期列向矩阵
        im_geotrans = dataset.GetGeoTransform() # 获取仿射矩阵信息
        im_proj = dataset.GetProjection() # 获取投影信息
        return im_data, nXSize, nYSize, im_geotrans, im_proj, coord_x, coord_y,month_arr,date_arr
    else:
        return im_data

def out_img(out_Path, im_data, rows, cols, im_geotrans, im_proj):
    driver = gdal.GetDriverByName("GTiff")
    data = driver.Create(out_Path, cols, rows, 1, gdal.GDT_Float32)
    data.SetGeoTransform(im_geotrans) #写入仿射变换参数
    data.SetProjection(im_proj) #写入投影
    data.GetRasterBand(1).WriteArray(im_data)
    
for month in range(6,7):
    month = str(month)
    str_month = month.zfill(2)
    
    model_path = r'K:\model\ERF\ERF_2019_smogn.pickle'
    root_dir = "K:\\CORRECT"#+str_month
    time = '2019'+str_month
    
    out_Path = 'K:\\result_tif\\ERF_smogn\\'+time
    if not os.path.exists(out_Path):
        os.makedirs(out_Path)
    
    
    
    
    #导入预先训练好的模型
    f = open(model_path,'rb') #注意此处model是rb
    s = f.read()
    model = pickle.loads(s)
    
    #获取所有数据的路径
    CO_path,HCHO_path,NO2_path,vieo_path,vino_path,sp_path,rh_path,blh_path,ssrd_path,t2m_path,u_path,v_path,alnid_path,cbh_path,AOD_path= get_tif_path(root_dir)#
    
    for f in range(len(rh_path)): 
        #获取单个arr
        rh_data, nXSize, nYSize, im_geotrans, im_proj, coord_x, coord_y,month_arr,date_arr = get_img(rh_path[f])
        
        CO_data = get_img(CO_path[f])
        HCHO_data = get_img(HCHO_path[f])
        NO2_data = get_img(NO2_path[f])
        AOD_data = get_img(AOD_path[f])
         
        alnid_data = get_img(alnid_path[f])
        vieo_data = get_img(vieo_path[f])
        vino_data = get_img(vino_path[f])
        sp_data = get_img(sp_path[f])
        blh_data = get_img(blh_path[f])
        cbh_data = get_img(cbh_path[f])
        ssrd_data = get_img(ssrd_path[f])
        t2m_data = get_img(t2m_path[f])
        u_data = get_img(u_path[f])
        v_data = get_img(v_path[f])
        
        #获取无效索引
        NO2_invalid = np.array([i for (i,v) in enumerate(NO2_data) if v<=0])
        CO_invalid = np.array([i for (i,v) in enumerate(CO_data) if v<=0])
        HCHO_invalid = np.array([i for (i,v) in enumerate(HCHO_data) if v<=0])
        
        
        
        #拼接arr
        #predict_X = np.hstack((im_month,date_arr,coord_y,coord_x,b_data,r_data,s_data,ssrd_data,t_data,tcc_data,tp_data,u_data, v_data,CO_data,HCHO_data,SO2_data,NO2_data))
        predict_X = np.hstack((month_arr,date_arr,coord_x, coord_y,CO_data,HCHO_data,NO2_data,AOD_data,
                               vieo_data,vino_data,sp_data,rh_data,blh_data,ssrd_data,t2m_data,u_data,v_data,cbh_data,alnid_data,))#
        
        
        
        
        #预测Y
        predict_Y = model.predict(predict_X)
        
        #无效值索引去除
        predict_Y[NO2_invalid] = 0
        predict_Y[CO_invalid] = 0
        predict_Y[HCHO_invalid] = 0
        #一维改二维，改为tif
        predict_Y = predict_Y.reshape((nYSize, nXSize))
        
        #创建文件名称
        outimg_path = out_Path +'\\'+rh_path[f].split('\\')[-1][5:13] + '_O3_GBRT' + '.tif'
        #写出tif
        out_img(outimg_path, predict_Y,nYSize, nXSize, im_geotrans, im_proj)
        print(outimg_path,'  prediction sucessful')


















    
    
    
