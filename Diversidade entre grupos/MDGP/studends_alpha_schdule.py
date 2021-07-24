from data.data_processing import Students
from mdgp import MDGP

students = Students()
students_in_class, distance_matrix = students.get_name_distance_matrix('all')
groups = 10

mdgp = MDGP(distance_matrix, groups)

cooling_schedule = mdgp.alpha_schedule(100000, 0.0001, 1000, 0.95)

mdgp.simulated_annealing_with_exec_log(cooling_schedule, 'all_students_alpha_schedule')
