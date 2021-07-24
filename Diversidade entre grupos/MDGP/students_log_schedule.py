from data.data_processing import Students
from mdgp import MDGP
import pandas as pd

students = Students()
students_in_class, distance_matrix = students.get_name_distance_matrix('all')
groups = 10

mdgp = MDGP(distance_matrix, groups)

cooling_schedule = mdgp.log_schedule(,0.001, 0.0001)

mdgp.simulated_annealing_with_exec_log(cooling_schedule, 'all_students_alpha_schedule')