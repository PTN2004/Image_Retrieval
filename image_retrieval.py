from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction

ROOT = 'data'
CLASS_NAMES = sorted(os.listdir(f"{ROOT}/train"))
embedding_function = OpenCLIPEmbeddingFunction()

def get_file_path(path):
    file_path = []
    for label in CLASS_NAMES:
        label_path = path + '/' + label
        for file in os.listdir(label_path):
            filepath = label_path + '/' + file
            file_path.append(filepath)
    return file_path

    
    
def create_vectordb(name_collection, file_path):
    db_client = chromadb.Client()
    collection = db_client.get_or_create_collection(name=name_collection)

    add_embedding(collection=collection, file_path=file_path)
    return collection

def get_single_image(image):
    embedding = embedding_function._encode_image(image=np.array(image))
    return embedding

def add_embedding(collection, file_path):
    ids = []    #id image
    embeddings = []
    for id, filepath in tqdm(enumerate(file_path)):
        print(f"Image_id = {id}, image_path: {filepath}")
        ids.append(f'id_{id}')
        image = Image.open(filepath)
        embedding = get_single_image(image = image)
        embeddings.append(embedding)
    collection.add(ids=ids, embeddings=embeddings)

def search(image_path, collection, n_results):
    query_image = Image.open(image_path)
    query_embedding = get_single_image(query_image)
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return results

def read_image_from_path(path, size):
    image = Image.open(path).convert("RGB").resize(size)
    return np.array(image)


def folder_to_images(folder, size):
    list_dir = [folder + '/' + name for name in os.listdir(folder)]
    image_array = np.zeros((len(list_dir), *size, 3))
    image_path = []
    for i, path in enumerate(list_dir):
        image_array[i] = read_image_from_path(path, size)
        image_path.append(path)

    return image_array, image_path


def absolute_difference(query, data):
    axis_batch_sizes = tuple(range(1, len(data.shape)))
    return np.sum(np.abs(data - query), axis=axis_batch_sizes)


def get_l1_score(root_image_path, query_path, size):
    query_image = read_image_from_path(query_path, size)
    embedding_image = get_single_image(query_image)
    ls_path_score = []
    for folder in os.listdir(root_image_path):
        if folder in CLASS_NAMES:
            path = root_image_path + folder
            image_array, image_path = folder_to_images(path, size)
            ls_embedding = []
            for index_img in range(len(image_array)):
                embedding = get_single_image(image_array[index_img].astype(np.uint8))
                ls_embedding.append(embedding)
            rates = absolute_difference(embedding_image, np.stack(ls_embedding))
            ls_path_score.extend(list(zip(image_path, rates)))
    return query_image, ls_path_score


def mean_square(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    return np.sum((data - query)**2, axis=axis_batch_size)


def get_l2_score(root_image_path, query_path, size):
    query_image = read_image_from_path(query_path, size=size)
    embedding_image = get_single_image(query_image)
    ls_path_score = []
    for folder in os.listdir(root_image_path):
        if folder in CLASS_NAMES:
            path = root_image_path + folder
            image_array, image_path = folder_to_images(path, size)
            embedding_list = []
            for i in range(len(image_array)):
                embedding = get_single_image(image=image_array[i].astype(np.uint8))
                embedding_list.append(embedding)
        
            rates = mean_square(embedding_image, np.stack(embedding_list))
            ls_path_score.extend(list(zip(image_path, rates)))

    return query_image, ls_path_score


def cosine_similarity(query, data):
    axis_batch_size = tuple(range(1, len(data.shape)))
    query_norm = np.sqrt(np.sum(query**2))
    data_norm = np.sqrt(np.sum(data**2, axis=axis_batch_size))

    return np.sum(data * query, axis=axis_batch_size) / (query_norm * data_norm + np.finfo(float).eps)

def get_cosine_similarity_score(root_image_path, query_path, size):
    query_image = read_image_from_path(query_path, size)
    embedding_image = get_single_image(query_image)
    ls_path_score = []
    for folder in os.listdir(root_image_path):
        if folder in CLASS_NAMES:
            path = root_image_path + folder
            image_array, image_path = folder_to_images(path, size)
            embedding_list = []
            for i in range(len(image_array)):
                embedding = get_single_image(image_array[i].astype(np.uint8))
                embedding_list.append(embedding)
            rates = cosine_similarity(embedding_image, embedding_list)
            ls_path_score.extend(list(zip(image_path, rates)))
    return query_image, ls_path_score


def plot_result(query_path, ls_path_score, size, reverse):
    fig = plt.figure(figsize=(15, 9))
    fig.add_subplot(2, 3, 1)
    plt.imshow(read_image_from_path(query_path, size))
    plt.title(f"Query path: {query_path.split('/')[2]}", fontsize=16)
    plt.axis('off')
    for i, path in enumerate(sorted(ls_path_score, key=lambda x: x[1], reverse=reverse)[:5], 2):
        fig.add_subplot(2, 3, i)
        plt.imshow(read_image_from_path(path[0], size=size))
        plt.title(f"Top {i-1}: {path[0].split('/')[2]}", fontsize=16)
        plt.axis('off')
    plt.show()

def plot_results(image_path, files_path, results):
    query_image = Image.open(image_path).resize((448,448))
    images = [query_image]
    class_name = []
    for id_img in results['ids'][0]:
        id_img = int(id_img.split('_')[-1])
        img_path = files_path[id_img]
        img = Image.open(img_path).resize((448,448))
        images.append(img)
        class_name.append(img_path.split('/')[2])

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Iterate through images and plot them
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        if i == 0:
            ax.set_title(f"Query Image: {image_path.split('/')[2]}")
        else:
            ax.set_title(f"Top {i+1}: {class_name[i-1]}")
        ax.axis('off')  # Hide axes
    # Display the plot
    plt.show()


root_image_path = f'{ROOT}/train/'
query_path = f"{ROOT}/test/Orange_easy/0_100.jpg"
test_path = f'{ROOT}/test/'
test_file_path = get_file_path(test_path)
test_file = test_file_path[1]
size = (448, 448)
image = Image.open(query_path)

file_path = get_file_path(root_image_path)
chroma_client = chromadb.Client()
l2_collection = chroma_client.get_or_create_collection(name="l2_collection", metadata={"hnsw:space": 'l2'})
add_embedding(collection=l2_collection, file_path=file_path)
results = search(test_file, l2_collection, 5)
plot_results(image_path=test_file, files_path=file_path, results=results)

