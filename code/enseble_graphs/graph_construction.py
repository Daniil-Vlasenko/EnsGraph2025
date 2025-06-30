import pandas
import numpy
import pickle
import gc
from os.path import dirname
from sklearn.linear_model import LogisticRegression
from os import mkdir
from shutil import rmtree
import paths    # local file containing data paths


def models_learning(files_in1, files_in2, files_out):
    """
    Training base models.

    :param files_in1: Tuple of two lists of edge feature files, i.e. Pearson correlations [[class1], [class2]].
    :param files_in2: Tuple of two lists of vertex feature files, i.e. means and standard deviations [[class1], [class2]].
    :param files_out: List of paths where the base trained models are saved.
    :return: None
    """
    assert len(files_in1[0]) == len(files_in1[1]) == len(files_in2[0]) == len(files_in2[1])
    data_edge1, data_edge2, data_vertex1, data_vertex2 = [], [], [], []
    for i in range(len(files_in1[0])):
        data_edge1.append(numpy.load(files_in1[0][i]))
        data_edge2.append(numpy.load(files_in1[1][i]))
        data_vertex1.append(pandas.read_pickle(files_in2[0][i])[['mean', 'std']].values)
        data_vertex2.append(pandas.read_pickle(files_in2[1][i])[['mean', 'std']].values)
    data_edge1, data_edge2, data_vertex1, data_vertex2 = (
        numpy.array(data_edge1), numpy.array(data_edge2),
        numpy.array(data_vertex1), numpy.array(data_vertex2)
    )
    count = 0
    for i in range(379):
        for j in range(i + 1, 379):
            if (i * 379 + j) % 10000 == 0:
                print(i * 379 + j)
            X = pandas.DataFrame({
                f'v_{i}_mean': numpy.concatenate([data_vertex1[:, i, 0], data_vertex2[:, i, 0]]),
                f'v_{j}_mean': numpy.concatenate([data_vertex1[:, j, 0], data_vertex2[:, j, 0]]),
                f'v_{i}_std': numpy.concatenate([data_vertex1[:, i, 1], data_vertex2[:, i, 1]]),
                f'v_{j}_std': numpy.concatenate([data_vertex1[:, j, 1], data_vertex2[:, j, 1]]),
                f'e_{i}_{j}': numpy.concatenate([data_edge1[:, i, j], data_edge2[:, i, j]])
            })
            Y = [1 for _ in range(data_vertex1.shape[0])] + [2 for _ in range(data_vertex1.shape[0])]

            model = LogisticRegression(random_state=0)
            model.fit(X, Y)
            with open(files_out[count], 'wb') as file:
                pickle.dump(model, file)
            count += 1


 def models_calculation(files_in1, files_in2, files_in3, folder_out):
    """
    Computation of ensemble graphs.

    :param files_in1: Tuple of two lists of edge feature files, i.e. Pearson correlations [[class1], [class2]].
    :param files_in2: Tuple of two lists of vertex feature files, i.e. means and standard deviations [[class1], [class2]].
    :param files_in3: List of paths to previously trained models.
    :param folder_out: Path to the folder where the constructed ensemble graphs are saved.
    :return: None
    """
    assert len(files_in1[0]) == len(files_in1[1]) == len(files_in2[0]) == len(files_in2[1])
    data_edge1, data_edge2, data_vertex1, data_vertex2 = [], [], [], []
    for i in range(len(files_in1[0])):
        data_edge1.append(numpy.load(files_in1[0][i]))
        data_edge2.append(numpy.load(files_in1[1][i]))
        data_vertex1.append(pandas.read_pickle(files_in2[0][i])[['mean', 'std']].values)
        data_vertex2.append(pandas.read_pickle(files_in2[1][i])[['mean', 'std']].values)
    data_edge1, data_edge2, data_vertex1, data_vertex2 = (
        numpy.array(data_edge1), numpy.array(data_edge2),
        numpy.array(data_vertex1), numpy.array(data_vertex2)
    )

    results1 = numpy.zeros(data_edge1.shape)
    results2 = numpy.zeros(data_edge1.shape)
    count = 0
    for i in range(379):
        for j in range(i + 1, 379):
            if (i * 379 + j) % 10000 == 0:
                print(i * 379 + j)
            X = pandas.DataFrame({
                f'v_{i}_mean': numpy.concatenate([data_vertex1[:, i, 0], data_vertex2[:, i, 0]]),
                f'v_{j}_mean': numpy.concatenate([data_vertex1[:, j, 0], data_vertex2[:, j, 0]]),
                f'v_{i}_std': numpy.concatenate([data_vertex1[:, i, 1], data_vertex2[:, i, 1]]),
                f'v_{j}_std': numpy.concatenate([data_vertex1[:, j, 1], data_vertex2[:, j, 1]]),
                f'e_{i}_{j}': numpy.concatenate([data_edge1[:, i, j], data_edge2[:, i, j]])
            })
            with open(files_in3[count], 'rb') as file:
                model = pickle.load(file)
            Y = model.predict_proba(X)
            results1[:, i, j] = Y[:data_edge1.shape[0], 1] - Y[:data_edge1.shape[0], 0]
            results1[:, j, i] = results1[:, i, j]
            results2[:, i, j] = Y[data_edge1.shape[0]:, 1] - Y[data_edge1.shape[0]:, 0]
            results2[:, j, i] = results2[:, i, j]
            count += 1

    for data1, data2, file1, file2 in zip(results1, results2, files_in1[0], files_in1[1]):
        numpy.save(f'{folder_out}/{file1.split('/')[-1]}', data1)
        numpy.save(f'{folder_out}/{file2.split('/')[-1]}', data2)


def cross_validation(files_in1, files_in2, encoding_type, models_files, folder_out1, folder_out2):
    """
    Cross-validation – training base models and constructing ensemble graphs for training and testing meta-models.

    :param files_in1: Tuple of two lists of edge feature files, i.e. Pearson correlations [[class1], [class2]].
    :param files_in2: Tuple of two lists of vertex feature files, i.e. means and standard deviations [[class1], [class2]].
    :param encoding_type: Data encoding type (file selection parameter for HCP data).
    :param models_files: List of paths to base models.
    :param folder_out1: Paths to folders where ensemble graphs for meta-model training are saved.
    :param folder_out2: Paths to folders where ensemble graphs for meta-model testing are saved.
    :return: None
    """
    for i in range(4):
        # Construct graphs for meta-model training
        for j in range(3):
            print(f'i: {i}, j: {j}')
            # Train base models
            edge1_tr, edge2_tr = paths.data_preparation(
                files_in1, encoding_type,
                paths.folds_ensemble[f'fold{i}'][f'fold{j}']['train']
            )
            vertex1_tr, vertex2_tr = paths.data_preparation(
                files_in2, encoding_type,
                paths.folds_ensemble[f'fold{i}'][f'fold{j}']['train']
            )
            rmtree(dirname(models_files[0]))
            mkdir(dirname(models_files[0]))
            models_learning([edge1_tr, edge2_tr], [vertex1_tr, vertex2_tr], models_files)

            # Construct graphs
            edge1_te, edge2_te = paths.data_preparation(
                files_in1, encoding_type,
                paths.folds_ensemble[f'fold{i}'][f'fold{j}']['test']
            )
            vertex1_te, vertex2_te = paths.data_preparation(
                files_in2, encoding_type,
                paths.folds_ensemble[f'fold{i}'][f'fold{j}']['test']
            )
            rmtree(dirname(models_files[0]))
            mkdir(dirname(models_files[0]))
            models_calculation(
                [edge1_te, edge2_te], [vertex1_te, vertex2_te],
                models_files, folder_out1[i][j]
            )
            gc.collect()

        # Construct graphs for meta-model testing
        # Train base models
        edge1_tr, edge2_tr = paths.data_preparation(
            files_in1, encoding_type,
            paths.folds_gnn[f'fold{i}']['train']
        )
        vertex1_tr, vertex2_tr = paths.data_preparation(
            files_in2, encoding_type,
            paths.folds_gnn[f'fold{i}']['train']
        )
        rmtree(dirname(models_files[0]))
        mkdir(dirname(models_files[0]))
        models_learning([edge1_tr, edge2_tr], [vertex1_tr, vertex2_tr], models_files)

        # Construct graphs
        edge1_te, edge2_te = paths.data_preparation(
            files_in1, encoding_type,
            paths.folds_gnn[f'fold{i}']['test']
        )
        vertex1_te, vertex2_te = paths.data_preparation(
            files_in2, encoding_type,
            paths.folds_gnn[f'fold{i}']['test']
        )
        rmtree(dirname(models_files[0]))
        mkdir(dirname(models_files[0]))
        models_calculation(
            [edge1_te, edge2_te], [vertex1_te, vertex2_te],
            models_files, folder_out2[i]
        )
        gc.collect()
