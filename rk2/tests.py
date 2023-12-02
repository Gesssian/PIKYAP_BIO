import unittest
from main import *


class rk2teste(unittest.TestCase):
    autoparks = [
        Autopark(1, 'Астафьево'),
        Autopark(2, 'Рязанский ЦАВ'),
        Autopark(3, 'Щёлковский АВ'),

        Autopark(11, 'Котельники'),
        Autopark(22, 'Рязань-2'),
        Autopark(33, 'Щёлково'),
    ]

    # Водители
    drivers = [
        Driver(1, 'Георгич', 25000, 1),
        Driver(2, 'Дмитрич', 35000, 2),
        Driver(3, 'Геннадич', 45000, 3),
        Driver(4, 'Палыч', 35000, 3),
        Driver(5, 'Михалыч', 25000, 3),
    ]

    goslings = [
        DriverInAutopark(1, 1),
        DriverInAutopark(2, 2),
        DriverInAutopark(3, 3),
        DriverInAutopark(3, 4),
        DriverInAutopark(3, 5),

        DriverInAutopark(11, 1),
        DriverInAutopark(22, 2),
        DriverInAutopark(33, 3),
        DriverInAutopark(33, 4),
        DriverInAutopark(33, 5),
    ]

    def test_one(self):
        one_to_many_fq = [(park.name, driver.fio, driver.sal)
                          for park in autoparks
                          for driver in drivers
                          if park.id == driver.park_id]
        self.assertEqual(ans_one(one_to_many_fq),
                         ['Астафьево', 'Георгич 25000'])

    def test_two(self):
        one_to_many_fq = [(park.name, driver.fio, driver.sal)
                          for park in autoparks
                          for driver in drivers
                          if park.id == driver.park_id]
        self.assertEqual(ans_two(one_to_many_fq),
                         [('Астафьево', 25000), ('Рязанский ЦАВ', 35000), ('Щёлковский АВ', 45000)])

    def test_three(self):
        one_to_many_curr = [(park.name, dia.park_id, dia.driver_id)
                            for park in autoparks
                            for dia in goslings
                            if park.id == dia.park_id]
        many_to_many_ans = [(park_name, d.fio)
                            for park_name, park_id, driver_id in one_to_many_curr
                            for d in drivers if d.id == driver_id]
        self.assertEqual(ans_three(many_to_many_ans),
                         {'Астафьево': ['Георгич'],
                          'Рязанский ЦАВ': ['Дмитрич'],
                          'Щёлковский АВ': ['Геннадич', 'Палыч', 'Михалыч'],
                          'Котельники': ['Георгич'], 'Рязань-2': ['Дмитрич'],
                          'Щёлково': ['Геннадич', 'Палыч', 'Михалыч']})


if __name__ == '__main__':
    unittest.main()