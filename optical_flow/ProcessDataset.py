# -*- coding: utf-8 -*-
"""
@author: Álefe Macedo
"""

import numpy as np
import os
from Optical_Flow_HS import calcOpticalFlowHS
import argparse

dtype = np.dtype([('path', str), ('class_name', str)])
# Caminho do diretório para salvar, de modo que este fique um nível abaixo do arquivo atual
saved_directory_of = os.path.join(os.path.split(os.getcwd())[0],'saved_dataset_of')
saved_directory_raw = os.path.join(os.path.split(os.getcwd())[0],'saved_dataset_raw')

def main():    
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--database", type=str, required=False, default="casia",
    	help="database name")
    ap.add_argument("-p", "--path", type=str, required=False, default="D:\CASIA_AR",
    	help="path to the database")
    args = vars(ap.parse_args())

    dataset_config = dict({
            'path': args['path'],
            'database': args['database'],
            'save_path_of': saved_directory_of,
            'save_path_raw': saved_directory_raw,
            'save_file_of': r'dataset_of.txt',
            'save_file_raw': r'dataset_raw.txt'})
    processDataset(dataset_config)

def processPath(path, database, save_path_of, save_path_raw):

    """
        Lista todos os arquivos ou diretórios na hierarquia atual
    """
    filenames = os.listdir(path)
    for filename in filenames:
        """
            Verifica se o filename atual é um diretório
        """
        if os.path.isdir(os.path.join(path, filename)):
            processPath(os.path.join(path, filename), database, save_path_of, save_path_raw)
        elif filename.lower().endswith(('.avi', '.mp4')):
            if database.lower() == 'casia':
                file_class = os.path.split(path)[1]
            elif database.lower() == 'weizmann':
                file_class = os.path.splitext(filename)[0].split('_')[1]

            """
                Constrói os caminhos para a classe do arquivo até os diretórios onde os datasets estão
                sendo salvos e verifica se os diretórios para esta classe já existem. Caso não existam,
                cria-se os diretórios
            """            
            class_path_of = os.path.join(save_path_of, file_class)
            class_path_raw = os.path.join(save_path_raw, file_class)
            
            if not (os.path.exists(class_path_of) and os.path.isdir(class_path_of)):
                try:
                    os.mkdir(class_path_of)
                except OSError:
                    print ("A criação do diretório %s falhou" % class_path_of)
                else:
                    print ("O diretório %s foi criado com sucesso" % class_path_of)
            
            if not (os.path.exists(class_path_raw) and os.path.isdir(class_path_raw)):
                try:
                    os.mkdir(class_path_raw)
                except OSError:
                    print ("A criação do diretório %s falhou" % class_path_raw)
                else:
                    print ("O diretório %s foi criado com sucesso" % class_path_raw)
            
            # Caminho até o arquivo
            video_path = os.path.join(path, filename)
            # Recupera o nome do vídeo sem a extensão
            video_name = os.path.splitext(filename)[0]

            calcOpticalFlowHS(video_path, video_name, class_path_of, class_path_raw)
        
def processDataset(dataset_config):
    
    
    """
        Verifica se já existem os diretórios para salvar os datasets raw e do optical flow,
        caso não existam cria-se estes.
    """
    if not os.path.exists(dataset_config['save_path_of']) or not os.path.isdir(dataset_config['save_path_of']):
        try:
            os.mkdir(dataset_config['save_path_of'])
        except OSError:
            print ("A criação do diretório %s falhou" % dataset_config['save_path_of'])
        else:
            print ("O diretório %s foi criado com sucesso" % dataset_config['save_path_of'])

    if not os.path.exists(dataset_config['save_path_raw']) or not os.path.isdir(dataset_config['save_path_raw']):
        try:
            os.mkdir(dataset_config['save_path_raw'])
        except OSError:
            print ("A criação do diretório %s falhou" % dataset_config['save_path_raw'])
        else:
            print ("O diretório %s foi criado com sucesso" % dataset_config['save_path_raw'])

    processPath(dataset_config['path'], dataset_config['database'], dataset_config['save_path_of'], dataset_config['save_path_raw'])

if __name__ == "__main__":
    main()
    print('[INFO] dataset processing finished...')         
