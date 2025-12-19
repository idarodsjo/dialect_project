import unittest
import dialect_mapper

# NB: these classes rely on a specific CSV. If the CSV is updated the cases may fail

class DialectMapperTests(unittest.TestCase):

    def test_get_old_municipalities_from_named_dialect(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            set(mm.get_old_municipalities_from_named_dialect('Sørlandsk')), 
            set(['Arendal', 'Bamble', 'Birkenes', 'Bygland', 'Bykle', 'Drangedal', 'Evje og Hornnes', 'Froland', 'Fyresdal', 'Gjerstad', 'Grimstad', 'Iveland', 'Kragerø', 'Kristiansand', 'Lillesand', 'Lindesnes', 'Mandal', 'Marnardal', 'Nissedal', 'Risør', 'Songdalen', 'Søgne', 'Tvedestrand', 'Valle', 'Vegårshei', 'Vennesla', 'Åmli'])
        )
    def test_get_new_municipalities_from_named_dialect(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            set(mm.get_new_municipalities_from_named_dialect('Sørlandsk')), 
            set(['Arendal', 'Bamble', 'Birkenes', 'Bygland', 'Bykle', 'Drangedal', 'Evje og Hornnes', 'Froland', 'Fyresdal', 'Gjerstad', 'Grimstad', 'Iveland', 'Kragerø', 'Kristiansand', 'Lillesand', 'Lindesnes', 'Nissedal', 'Risør', 'Tvedestrand', 'Valle', 'Vegårshei', 'Vennesla', 'Åmli'])
        )
    def test_get_old_counties_from_named_dialect(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            set(mm.get_old_counties_from_named_dialect('Sørlandsk')), 
            set(['Aust-Agder', 'Vest-Agder', 'Telemark'])
        )
    def test_get_new_counties_from_named_dialect(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            set(mm.get_new_counties_from_named_dialect('Sørlandsk')), 
            set(['Agder', 'Vestfold og Telemark'])
        )

    def test_get_named_dialect_by_old_municipality(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_named_dialect_by_old_municipality('Kristiansand'), 
            ['Sørlandsk']
        )
    def test_get_named_dialect_by_new_municipality(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_named_dialect_by_new_municipality('Kristiansand'), 
            ['Sørlandsk']
        )
    def test_get_named_dialect_by_old_county(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_named_dialect_by_old_county('Aust-Agder'), 
            ['Sørlandsk']
        )
    def test_get_named_dialect_by_new_county(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_named_dialect_by_new_county('Agder'), 
            ['Sørlandsk', 'Sørvestlandsk']
        )
    
    def test_get_named_dialect_test_old_municipality(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_named_dialect('Songdalen'),
            'Sørlandsk'
        )
    def test_get_named_dialect_test_new_municipality(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_named_dialect('Kristiansand'),
            'Sørlandsk'
        )
    def test_get_named_dialect_test_old_county(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_named_dialect('Aust-Agder'),
            'Sørlandsk'
        )
    def test_get_named_dialect_test_new_county(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_named_dialect('Agder'),
            ['Sørlandsk', 'Sørvestlandsk']
        )
    def test_get_named_dialect_test_bad_input(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_named_dialect('Seattle'),
            None
        )


    def test_get_old_municipalities_from_numeric_dialect(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            set(mm.get_old_municipalities_from_numeric_dialect(17)), 
            set(['', 'Askøy', 'Bergen', 'Fjell', 'Os', 'Sund'])
        )
    def test_get_new_municipalities_from_numeric_dialect(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            set(mm.get_new_municipalities_from_numeric_dialect(17)), 
            set(['Askøy', 'Bergen', 'Bjørnafjorden', 'Øygarden'])
        )
    def test_get_old_counties_from_numeric_dialect(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            set(mm.get_old_counties_from_numeric_dialect('17')), 
            set(['Hordaland'])
        )
    def test_get_new_counties_from_numeric_dialect(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            set(mm.get_new_counties_from_numeric_dialect('17')), 
            set(['Vestland'])
        )

    def test_get_numeric_dialect_by_old_municipality(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_numeric_dialect_by_old_municipality('Kristiansand'), 
            ['19']
        )
    def test_get_numeric_dialect_by_new_municipality(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_numeric_dialect_by_new_municipality('Kristiansand'), 
            ['19']
        )
    def test_get_numeric_dialect_by_old_county(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_numeric_dialect_by_old_county('Aust-Agder'), 
            ['20']
        )
    def test_get_numeric_dialect_by_new_county(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_numeric_dialect_by_new_county('Agder'), 
            ['19', '20']
        )
    
    def test_get_numeric_dialect_test_old_municipality(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_numeric_dialect('Songdalen'),
            '19'
        )
    def test_get_numeric_dialect_test_new_municipality(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_numeric_dialect('Kristiansand'),
            '19'
        )
    def test_get_numeric_dialect_test_old_county(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_numeric_dialect('Aust-Agder'),
            '20'
        )
    def test_get_numeric_dialect_test_new_county(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_numeric_dialect('Agder'),
            ['19', '20']
        )
    def test_get_numeric_dialect_test_bad_input(self):
        mm = dialect_mapper.mapper_methods()
        self.assertEqual(
            mm.get_numeric_dialect('Seattle'),
            None
        )

    def test_nbtale_mapper_corrections(self):
        mm = dialect_mapper.mapper_methods()
        mm.enable_nbtale_corrections()
        named_dialect = mm.get_named_dialect('hyllestand')
        self.assertEqual(
            named_dialect,
            'Nordvestlandsk'
        )

    def test_npsc_mapper_corrections(self):
        mm = dialect_mapper.mapper_methods()
        mm.enable_npsc_corrections()
        named_dialect = mm.get_named_dialect('Vestfossen')
        self.assertEqual(
            named_dialect,
            'Østlandsk'
        )

    def test_npsc_mapper_corrections_notNorway(self):
        mm = dialect_mapper.mapper_methods()
        mm.enable_npsc_corrections()
        named_dialect = mm.get_named_dialect('Tehran')
        self.assertEqual(
            named_dialect,
            None
        )

    def test_nbtale_mapper_corrections_herøy(self):
        mm = dialect_mapper.mapper_methods()
        mm.enable_nbtale_corrections()
        named_dialect = mm.get_named_dialect('Herøy')
        self.assertEqual(
            named_dialect,
            None
        )
    
    def test_nbtale_named_dialect_from_id_mod1(self):
        mm = dialect_mapper.mapper_methods()
        named_dialect = mm.get_nbtale_named_dialect_from_id('p1_g02_f2_4')
        self.assertEqual(
            named_dialect,
            'Troms-Finnmarks-mål'
        )
    
    def test_nbtale_named_dialect_from_id_mod2(self):
        mm = dialect_mapper.mapper_methods()
        named_dialect = mm.get_nbtale_named_dialect_from_id('p2_g14_m0_5_t')
        self.assertEqual(
            named_dialect,
            'Utenlandsk'
        )

    def test_nbtale_named_dialect_from_id_mod3(self):
        mm = dialect_mapper.mapper_methods()
        named_dialect = mm.get_nbtale_named_dialect_from_id('p2_g24_n0_5_t')
        self.assertEqual(
            named_dialect,
            'Nordvestlandsk'
        )

    def test_stortinget_mapper_corrections_fana(self):
        mm = dialect_mapper.mapper_methods()
        mm.enable_stortinget_corrections()
        named_diaelct = mm.get_named_dialect('Fana')
        self.assertEqual(
            named_diaelct,
            'Sørvestlandsk'
        )

    def test_cardinal_dialect_from_dia(self):
        mm = dialect_mapper.mapper_methods()
        card_dia = mm.get_cardinal_dialect('Sørlandsk')
        self.assertEqual(
            card_dia,
            'south'
        )

    def test_cardinal_dialect_from_kommune(self):
        mm = dialect_mapper.mapper_methods()
        card_dia = mm.get_cardinal_dialect('Trondheim')
        self.assertEqual(
            card_dia,
            'mid'
        )
    
    def test_cardinal_dialect_from_badInput(self):
        mm = dialect_mapper.mapper_methods()
        card_dia = mm.get_cardinal_dialect('Seattle')
        self.assertEqual(
            card_dia,
            None
        )

if __name__ == "__main__":
    unittest.main()