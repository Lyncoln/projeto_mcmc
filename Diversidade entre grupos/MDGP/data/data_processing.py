import pandas as pd
import jellyfish as jf
import numpy as np
from typing import List


class Students:
    def __init__(self) -> None:
        self.students = pd.read_csv('data/students.csv', index_col=0)

    def get_available_classes(self) -> List:
        return list(self.students.sort_values(by='turma')['turma'].unique())

    def get_name_distance_matrix(self, student_class: str) -> np.array:
        students = self.students.copy()
        students = students.loc[students['turma'] == student_class]

        students_combs = students.pivot_table(index='nome_aluno', columns='nome_aluno', values=None)
        students_combs.columns = students_combs.columns.droplevel()
        students_combs.index.name = None
        students_combs = students_combs.melt(ignore_index=False).reset_index().drop(columns='value').rename(columns={'index': 'student1', 'nome_aluno': 'student2'})
        students_combs = students_combs.sort_values(by=['student1', 'student2'])
        students_combs.loc[:, 'pair_distance'] = students_combs.apply(lambda x: jf.levenshtein_distance(x['student1'], x['student2']), axis=1)
        students_matrix = students_combs.pivot_table(index='student1', columns='student2', values='pair_distance').values

        return students, students_matrix
