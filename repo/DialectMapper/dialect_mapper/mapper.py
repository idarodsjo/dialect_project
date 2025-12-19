"""
A class to help mapping between Norsk kommuner and dialekt names
"""

import csv
import sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import mapping_data

from collections import namedtuple

# A bunch of methods that makes querying dialectal relationships easier
# Code created by Phoebe Parsons on Jan 14 2022
# the CSV file was created by Phoebe Parsons via reading dialectal maps,
# scraping Wikipedia, and manual correction (with feedback from Per Erik Solberg)

## EDIT: screw it, we're using Pandas 
# Many of these methods could be improved via the use of Pandas. But, I 
# didn't want to force any other potential users to install Pandas as well

class mapper_methods:
    # ----------------- Disambiguation methods -----------------
    def is_ambiguious_municipality(self, municipality: str) -> bool:
        # ensure we don't have municipalities with the same name but different dialects
        old_dialects = self.get_named_dialect_by_old_municipality(municipality)
        new_dialects = self.get_named_dialect_by_new_municipality(municipality)
        if old_dialects != new_dialects:
            if old_dialects == [] or new_dialects == []:
                return False
            return True
        return False

    def is_ambiguious_county(self, municipality: str) -> bool:
        # ensure we don't have municipalities with the same name but different dialects
        if self.get_named_dialect_by_old_municipality(municipality) == self.get_named_dialect_by_new_municipality(municipality):
            return False
        return True

    def format_dialect_response(self, dialects):
        if self.collapse_fine_grained_dialects:
            dialects = [self._collapse_fine_granded_dialects(d) for d in dialects]
        if len(dialects) == 1:
            dialects = dialects[0]
        return dialects

    # ----------------- CARDINAL dialect methods -----------------
    def get_cardinal_dialect(self, input_str: str) -> str:
        # we'll allow the input to either be a dialect region or a kommune/fylke
        fine_to_cardinal = {
            'Østlandsk' : 'east',
            'Midlandsk' : 'east',
            'Namdalsk' : 'mid',
            'Østtrøndsk' : 'mid',
            'Uttrøndersk' : 'mid',
            'Trøndsk' : 'mid',
            'Helgelandsk' : 'north',
            'Nordlandsk' : 'north',
            'Troms-Finnmarks-mål' : 'north',
            'Sørlandsk' : 'south',
            'Sørvestlandsk' : 'west',
            'Nordvestlandsk': 'west'
        }
        if input_str in fine_to_cardinal:
            return fine_to_cardinal[input_str]
        else:
            raise Exception('This method is depreciated for input other than named dialects. {} does not appear to be a named dialect. Please use either get_cardinal_four_dialect() or get_cardinal_five_dialect()'.format(input_str))
            # if isinstance(input_str, str):
            #     named_dialect = self.get_named_dialect(input_str)
            #     if named_dialect and named_dialect in fine_to_cardinal:
            #         return fine_to_cardinal[named_dialect]
            #     else:
            #         return None
            # elif isinstance(input_str, int):
            #     named_dialect = self.get

    # ----------------- CARDINAL dialect methods -----------------
    def get_cardinal_four_by_old_municipality(self, old_municipality) -> list:
        old_municipality = old_municipality.lower().strip()
        old_municipality = self._get_corrections(old_municipality)
        # if old_municipality is not a Norwegian muni then it will be none
        # which causes problems b/c in the csv data old_muni being empty means there isn't an old muni corresponding to the new muni
        # thus we want to ignore old_munis being none instead of returning all the new munis w/o an old 
        if old_municipality == '':
            return []
        return sorted(list(set([x.cardinal_four for x in self.csv_tuples if x.old_muni.lower().strip() == old_municipality])))
    def get_cardinal_four_by_new_municipality(self, new_municipality) -> list:
        new_municipality = new_municipality.lower().strip()
        new_municipality = self._get_corrections(new_municipality)
        return sorted(list(set([x.cardinal_four for x in self.csv_tuples if x.new_muni.lower().strip() == new_municipality])))
    def get_cardinal_four_by_old_county(self, old_county) -> list:
        return sorted(list(set([x.cardinal_four for x in self.csv_tuples if x.old_county.lower().strip() == old_county.lower().strip()])))
    def get_cardinal_four_by_new_county(self, new_county) -> list:
        return sorted(list(set([x.cardinal_four for x in self.csv_tuples if x.new_county.lower().strip() == new_county.lower().strip()])))
    def get_cardinal_four_by_new_county_2024(self, new_county) -> list:
        return sorted(list(set([x.cardinal_four for x in self.csv_tuples if x.new_county_2024.lower().strip() == new_county.lower().strip()])))
    
    def get_cardinal_four(self, lookup_by: str, resolve_ambigious='new'):
        if self.is_ambiguious_municipality(lookup_by):
            resolve_ambigious = resolve_ambigious.lower().strip()
            if resolve_ambigious in ['new', 'old']:
                if resolve_ambigious == 'new':
                    dialects = self.get_cardinal_four_by_new_municipality(lookup_by)
                    return self.format_dialect_response(dialects)
                else:
                    dialects = self.get_cardinal_four_by_old_municipality(lookup_by)
                    return self.format_dialect_response(dialects)
            else:
                print("Unknown way of resolving ambigious municipality for {}. Using new municipality.".format(lookup_by))

        # by looking up by old municipality first we're prioritizing it. I don't have a super strong arguement as to the why,
        # presumably old municipalities will have fewer one to many mappings. But, if we feel like going with the new municipalities
        # is better this can easily be changed                
        dialects = self.get_cardinal_four_by_old_municipality(lookup_by)
        if len(dialects) > 0:
            return self.format_dialect_response(dialects)
        else:
            dialects = self.get_cardinal_four_by_new_municipality(lookup_by)
            if len(dialects) > 0:
                return self.format_dialect_response(dialects)
            else:
                dialects = self.get_cardinal_four_by_old_county(lookup_by)
                if len(dialects) > 0:
                    return self.format_dialect_response(dialects)
                else:
                    dialects = self.get_cardinal_four_by_new_county(lookup_by)
                    if len(dialects) > 0:
                        return self.format_dialect_response(dialects)
                    else:
                        dialects = self.get_cardinal_four_by_new_county_2024(lookup_by)
                        if len(dialects) > 0:
                            return self.format_dialect_response(dialects)
                        else:
                            print("ERROR: cannot find named dialect for: {}".format(lookup_by))
                            return None
                        
    def get_cardinal_five_by_old_municipality(self, old_municipality) -> list:
        old_municipality = old_municipality.lower().strip()
        old_municipality = self._get_corrections(old_municipality)
        # if old_municipality is not a Norwegian muni then it will be none
        # which causes problems b/c in the csv data old_muni being empty means there isn't an old muni corresponding to the new muni
        # thus we want to ignore old_munis being none instead of returning all the new munis w/o an old 
        if old_municipality == '':
            return []
        return sorted(list(set([x.cardinal_five for x in self.csv_tuples if x.old_muni.lower().strip() == old_municipality])))
    def get_cardinal_five_by_new_municipality(self, new_municipality) -> list:
        new_municipality = new_municipality.lower().strip()
        new_municipality = self._get_corrections(new_municipality)
        return sorted(list(set([x.cardinal_five for x in self.csv_tuples if x.new_muni.lower().strip() == new_municipality])))
    def get_cardinal_five_by_old_county(self, old_county) -> list:
        return sorted(list(set([x.cardinal_five for x in self.csv_tuples if x.old_county.lower().strip() == old_county.lower().strip()])))
    def get_cardinal_five_by_new_county(self, new_county) -> list:
        return sorted(list(set([x.cardinal_five for x in self.csv_tuples if x.new_county.lower().strip() == new_county.lower().strip()])))
    def get_cardinal_five_by_new_county_2024(self, new_county) -> list:
        return sorted(list(set([x.cardinal_five for x in self.csv_tuples if x.new_county_2024.lower().strip() == new_county.lower().strip()])))
    
    def get_cardinal_five(self, lookup_by: str, resolve_ambigious='new'):
        if self.is_ambiguious_municipality(lookup_by):
            resolve_ambigious = resolve_ambigious.lower().strip()
            if resolve_ambigious in ['new', 'old']:
                if resolve_ambigious == 'new':
                    dialects = self.get_cardinal_five_by_new_municipality(lookup_by)
                    return self.format_dialect_response(dialects)
                else:
                    dialects = self.get_cardinal_five_by_old_municipality(lookup_by)
                    return self.format_dialect_response(dialects)
            else:
                print("Unknown way of resolving ambigious municipality for {}. Using new municipality.".format(lookup_by))

        # by looking up by old municipality first we're prioritizing it. I don't have a super strong arguement as to the why,
        # presumably old municipalities will have fewer one to many mappings. But, if we feel like going with the new municipalities
        # is better this can easily be changed                
        dialects = self.get_cardinal_five_by_old_municipality(lookup_by)
        if len(dialects) > 0:
            return self.format_dialect_response(dialects)
        else:
            dialects = self.get_cardinal_five_by_new_municipality(lookup_by)
            if len(dialects) > 0:
                return self.format_dialect_response(dialects)
            else:
                dialects = self.get_cardinal_five_by_old_county(lookup_by)
                if len(dialects) > 0:
                    return self.format_dialect_response(dialects)
                else:
                    dialects = self.get_cardinal_five_by_new_county(lookup_by)
                    if len(dialects) > 0:
                        return self.format_dialect_response(dialects)
                    else:
                        dialects = self.get_cardinal_five_by_new_county_2024(lookup_by)
                        if len(dialects) > 0:
                            return self.format_dialect_response(dialects)
                        else:
                            print("ERROR: cannot find named dialect for: {}".format(lookup_by))
                            return None


    # ----------------- NAMED dialect methods -----------------
    def get_old_municipalities_from_named_dialect(self, named_dialect: str) -> list:
        return sorted(list(set([x.old_muni for x in self.csv_tuples if x.named_dialect.lower().strip() == named_dialect.lower().strip()])))
    def get_new_municipalities_from_named_dialect(self, named_dialect: str) -> list:
        return sorted(list(set([x.new_muni for x in self.csv_tuples if x.named_dialect.lower().strip() == named_dialect.lower().strip()])))
    def get_old_counties_from_named_dialect(self, named_dialect: str) -> list:
        return sorted(list(set([x.old_county for x in self.csv_tuples if x.named_dialect.lower().strip() == named_dialect.lower().strip()])))
    def get_new_counties_from_named_dialect(self, named_dialect: str) -> list:
        return sorted(list(set([x.new_county for x in self.csv_tuples if x.named_dialect.lower().strip() == named_dialect.lower().strip()])))
    def get_new_counties_2024_from_named_dialect(self, named_dialect: str) -> list:
        return sorted(list(set([x.new_county_2024 for x in self.csv_tuples if x.named_dialect.lower().strip() == named_dialect.lower().strip()])))

    def get_named_dialect_by_old_municipality(self, old_municipality) -> list:
        old_municipality = old_municipality.lower().strip()
        old_municipality = self._get_corrections(old_municipality)
        # if old_municipality is not a Norwegian muni then it will be none
        # which causes problems b/c in the csv data old_muni being empty means there isn't an old muni corresponding to the new muni
        # thus we want to ignore old_munis being none instead of returning all the new munis w/o an old 
        if old_municipality == '':
            return []
        return sorted(list(set([x.named_dialect for x in self.csv_tuples if x.old_muni.lower().strip() == old_municipality])))
    def get_named_dialect_by_new_municipality(self, new_municipality) -> list:
        new_municipality = new_municipality.lower().strip()
        new_municipality = self._get_corrections(new_municipality)
        return sorted(list(set([x.named_dialect for x in self.csv_tuples if x.new_muni.lower().strip() == new_municipality])))
    def get_named_dialect_by_old_county(self, old_county) -> list:
        return sorted(list(set([x.named_dialect for x in self.csv_tuples if x.old_county.lower().strip() == old_county.lower().strip()])))
    def get_named_dialect_by_new_county(self, new_county) -> list:
        return sorted(list(set([x.named_dialect for x in self.csv_tuples if x.new_county.lower().strip() == new_county.lower().strip()])))
    def get_named_dialect_by_new_county_2024(self, new_county) -> list:
        return sorted(list(set([x.named_dialect for x in self.csv_tuples if x.new_county_2024.lower().strip() == new_county.lower().strip()])))
    
    def get_named_dialect(self, lookup_by: str, resolve_ambigious='new'):
        if self.is_ambiguious_municipality(lookup_by):
            resolve_ambigious = resolve_ambigious.lower().strip()
            if resolve_ambigious in ['new', 'old']:
                if resolve_ambigious == 'new':
                    dialects = self.get_named_dialect_by_new_municipality(lookup_by)
                    return self.format_dialect_response(dialects)
                else:
                    dialects = self.get_named_dialect_by_old_municipality(lookup_by)
                    return self.format_dialect_response(dialects)
            else:
                print("Unknown way of resolving ambigious municipality for {}. Using new municipality.".format(lookup_by))

        # by looking up by old municipality first we're prioritizing it. I don't have a super strong arguement as to the why,
        # presumably old municipalities will have fewer one to many mappings. But, if we feel like going with the new municipalities
        # is better this can easily be changed                
        dialects = self.get_named_dialect_by_old_municipality(lookup_by)
        if len(dialects) > 0:
            return self.format_dialect_response(dialects)
        else:
            dialects = self.get_named_dialect_by_new_municipality(lookup_by)
            if len(dialects) > 0:
                return self.format_dialect_response(dialects)
            else:
                dialects = self.get_named_dialect_by_old_county(lookup_by)
                if len(dialects) > 0:
                    return self.format_dialect_response(dialects)
                else:
                    dialects = self.get_named_dialect_by_new_county(lookup_by)
                    if len(dialects) > 0:
                        return self.format_dialect_response(dialects)
                    else:
                        dialects = self.get_named_dialect_by_new_county_2024(lookup_by)
                        if len(dialects) > 0:
                            return self.format_dialect_response(dialects)
                        else:
                            print("ERROR: cannot find named dialect for: {}".format(lookup_by))
                            return None

    def get_nbtale_named_dialect_from_id(self, speaker_id: str):
        """ Manual work was done to create a speaker ID to dialect mapping for NB Tale speakers
            The bulk of the dialects where assigned using this code. However, I (Phoebe) also went
            through and did my best to add dialects where the kommune was missing but other metadata
            allowed me to make a guess at the dialect

        Args:
            speaker_id (str): A speaker ID in the NB Tale format (e.g. "p2_g24_f0_1_t)

        Returns:
            str: The name of the dialect region for that speaker
        """
        if len(self.nbtale_speakers_to_named_dialects) == 0:
            for module_number in ['1', '2', '3']:
                c_reader = csv.reader(
                    StringIO(
                        pkg_resources.read_text(
                            mapping_data, 
                            f'NB_Tale_Informantdata_module_{module_number}_updated.csv'
                        )
                    )
                )
                for row in c_reader:
                    csv_speaker_id = row[0]
                    if csv_speaker_id != 'Informant-ID':
                        # NOTE this will break if the format of the file changes!
                        # (e.g. if another column is added)
                        self.nbtale_speakers_to_named_dialects[csv_speaker_id] = row[-1]
        if speaker_id in self.nbtale_speakers_to_named_dialects:
            return self.nbtale_speakers_to_named_dialects[speaker_id]
        return ''

    # ----------------- NUMERIC dialect methods -----------------
    def get_old_municipalities_from_numeric_dialect(self, numeric_dialect: str) -> list:
        return sorted(list(set([x.old_muni for x in self.csv_tuples if int(x.numeric_dialect) == int(numeric_dialect)])))
    def get_new_municipalities_from_numeric_dialect(self, numeric_dialect: str) -> list:
        return sorted(list(set([x.new_muni for x in self.csv_tuples if int(x.numeric_dialect) == int(numeric_dialect)])))
    def get_old_counties_from_numeric_dialect(self, numeric_dialect: str) -> list:
        return sorted(list(set([x.old_county for x in self.csv_tuples if int(x.numeric_dialect) == int(numeric_dialect)])))
    def get_new_counties_from_numeric_dialect(self, numeric_dialect: str) -> list:
        return sorted(list(set([x.new_county for x in self.csv_tuples if int(x.numeric_dialect) == int(numeric_dialect)])))

    def get_numeric_dialect_by_old_municipality(self, old_municipality) -> list:
        old_municipality = old_municipality.lower().strip()
        old_municipality = self._get_corrections(old_municipality)
        # if old_municipality is not a Norwegian muni then it will be none
        # which causes problems b/c in the csv data old_muni being empty means there isn't an old muni corresponding to the new muni
        # thus we want to ignore old_munis being none instead of returning all the new munis w/o an old 
        if old_municipality == '':
            return []
        return sorted(list(set([x.numeric_dialect for x in self.csv_tuples if x.old_muni.lower().strip() == old_municipality])))
    def get_numeric_dialect_by_new_municipality(self, new_municipality) -> list:
        new_municipality = new_municipality.lower().strip()
        new_municipality = self._get_corrections(new_municipality)
        return sorted(list(set([x.numeric_dialect for x in self.csv_tuples if x.new_muni.lower().strip() == new_municipality])))
    def get_numeric_dialect_by_old_county(self, old_county) -> list:
        return sorted(list(set([x.numeric_dialect for x in self.csv_tuples if x.old_county.lower().strip() == old_county.lower().strip()])))
    def get_numeric_dialect_by_new_county(self, new_county) -> list:
        return sorted(list(set([x.numeric_dialect for x in self.csv_tuples if x.new_county.lower().strip() == new_county.lower().strip()])))
    def get_numeric_dialect_by_new_county_2024(self, new_county) -> list:
        return sorted(list(set([x.numeric_dialect for x in self.csv_tuples if x.new_county_2024.lower().strip() == new_county.lower().strip()])))
    
    def get_numeric_dialect(self, lookup_by: str, resolve_ambigious='new'):
        if self.is_ambiguious_municipality(lookup_by):
            resolve_ambigious = resolve_ambigious.lower().strip()
            if resolve_ambigious in ['new', 'old']:
                if resolve_ambigious == 'new':
                    dialects = self.get_numeric_dialect_by_new_municipality(lookup_by)
                    return self.format_dialect_response(dialects)
                else:
                    dialects = self.get_numeric_dialect_by_old_municipality(lookup_by)
                    return self.format_dialect_response(dialects)
            else:
                print("Unknown way of resolving ambigious municipality for {}. Using new municipality.".format(lookup_by))

        # by looking up by old municipality first we're prioritizing it. I don't have a super strong arguement as to the why,
        # presumably old municipalities will have fewer one to many mappings. But, if we feel like going with the new municipalities
        # is better this can easily be changed
        dialects = self.get_numeric_dialect_by_old_municipality(lookup_by)
        if len(dialects) > 0:
            return self.format_dialect_response(dialects)
        else:
            dialects = self.get_numeric_dialect_by_new_municipality(lookup_by)
            if len(dialects) > 0:
                return self.format_dialect_response(dialects)
            else:
                dialects = self.get_numeric_dialect_by_old_county(lookup_by)
                if len(dialects) > 0:
                    return self.format_dialect_response(dialects)
                else:
                    dialects = self.get_numeric_dialect_by_new_county(lookup_by)
                    if len(dialects) > 0:
                        return self.format_dialect_response(dialects)
                    else:
                        dialects = self.get_numeric_dialect_by_new_county_2024(lookup_by)
                        if len(dialects) > 0:
                            return self.format_dialect_response(dialects)
                        else:
                            print("ERROR: cannot find numeric dialect for: {}".format(lookup_by))
                            return None

    def enable_nbtale_corrections(self, ignore_herøy=True) -> None:
        # NBTale has some human errors in the kommune names. I've created a mapping from the NB Tale names to what they should be
        # this method will switch the flag so later queries use the corrected mapping and load the mapping data
        # NOTE: this will only correct the input. Presumably the resource table has the correct names
        # HERØY ADDITION: There are 2 kommuner with this name. We cannot reasonably tell them apart. We can ignore the 1 speaker from here
        self.use_nbtale_corrections = True
        cReader = csv.reader(
            StringIO(
                pkg_resources.read_text(
                    mapping_data, 
                    'nbtale_transform.csv'
                )
            )
        )
        for row in cReader:
            self.nbtale_corrections[row[0]] = row[1]
        if ignore_herøy:
            self.nbtale_corrections['herøy'] = ''

    def enable_npsc_corrections(self) -> None:
        # There are some place_of_birth's in NPSC that are either cities/towns instead of communes or are places outside of Norway
        # this method will switch the flag so later queries use the corrected mapping and load the mapping data
        # NOTE: this will only correct the input. Presumably the resource table has the correct names
        self.use_npsc_corrections = True
        cReader = csv.reader(
            StringIO(
                pkg_resources.read_text(
                    mapping_data, 
                    'npsc_transform.csv'
                )
            )
        )
        for row in cReader:
            self.npsc_corrections[row[0]] = row[1]

    def enable_stortinget_corrections(self) -> None:
        # There are some birth_kommunes from the Stortinget API that are either cities/towns instead of kommunes or have a typo ("opdal" looking at you)
        # this method will switch the flag so later queries use the corrected mapping and load the mapping data
        # NOTE: this will only correct the input. Presumably the resource table has the correct names
        self.use_stortinget_corrections = True
        cReader = csv.reader(
            StringIO(
                pkg_resources.read_text(
                    mapping_data, 
                    'stortinget_transform.csv'
                )
            )
        )
        for row in cReader:
            self.stortinget_corrections[row[0]] = row[1]

    def enable_ndc_corrections(self) -> None:
        # There are some birth_kommunes from the Stortinget API that are either cities/towns instead of kommunes or have a typo ("opdal" looking at you)
        # this method will switch the flag so later queries use the corrected mapping and load the mapping data
        # NOTE: this will only correct the input. Presumably the resource table has the correct names
        self.use_ndc_corrections = True
        cReader = csv.reader(
            StringIO(
                pkg_resources.read_text(
                    mapping_data, 
                    'ndc_transform.csv'
                )
            )
        )
        for row in cReader:
            self.ndc_corrections[row[0].lower()] = row[1].lower()

    def _get_nbtale_correction(self, lookup_by: str) -> str:
        if lookup_by in self.nbtale_corrections:
            lookup_by = self.nbtale_corrections[lookup_by]
        return lookup_by

    def _get_npsc_correction(self, lookup_by: str) -> str:
        if lookup_by in self.npsc_corrections:
            lookup_by = self.npsc_corrections[lookup_by]
        return lookup_by
    
    def _get_stortinget_correction(self, lookup_by: str) -> str:
        if lookup_by in self.stortinget_corrections:
            lookup_by = self.stortinget_corrections[lookup_by]
        return lookup_by
    
    def _get_ndc_corrections(self, lookup_by: str) -> str:
        if lookup_by in self.ndc_corrections:
            lookup_by = self.ndc_corrections[lookup_by]
        return lookup_by

    def _get_corrections(self, lookup_by: str) -> str:
        if self.use_nbtale_corrections:
            lookup_by = self._get_nbtale_correction(lookup_by)
        if self.use_npsc_corrections:
            lookup_by = self._get_npsc_correction(lookup_by)
        if self.use_stortinget_corrections:
            lookup_by = self._get_stortinget_correction(lookup_by)
        if self.use_ndc_corrections:
            lookup_by = self._get_ndc_corrections(lookup_by)
        return lookup_by

    def enable_fine_grained_dialect_collapse(self):
        self.collapse_fine_grained_dialects = True
    def disable_fine_grained_dialect_collapse(self):
        self.collapse_fine_grained_dialects = False

    def _collapse_fine_granded_dialects(self, dialect):
        if dialect in ['Østtrøndsk', 'Namdalsk', 'Uttrøndersk']:
            return 'Trøndsk'
        elif dialect == 'Midlandsk':
            return 'Østlandsk'
        else:
            return dialect

    def __init__(self) -> None:
        self.use_nbtale_corrections = False
        self.use_npsc_corrections = False
        self.use_stortinget_corrections = False
        self.use_ndc_corrections = False
        self.raw_csv_data = []
        self.nbtale_corrections = {}
        self.nbtale_speakers_to_named_dialects = {}
        self.npsc_corrections = {}
        self.stortinget_corrections = {}
        self.ndc_corrections = {}
        self.collapse_fine_grained_dialects = False

        cReader = csv.reader(
            StringIO(
                pkg_resources.read_text(
                    mapping_data, 
                    # 'muni_county_namedDialect_numericDialect_mapping_manual_additions_renamed_2024.csv'
                    'muni_county_namedDialect_numericDialect_mapping_manual_additions_renamed_2024_cardinals.csv'
                )
            )
        )
        for row in cReader:
            self.raw_csv_data.append(row)
        headers = self.raw_csv_data.pop(0)
        csv_row_tuple = namedtuple('csv_row_tuple', headers)
        self.csv_tuples = [csv_row_tuple(*row) for row in self.raw_csv_data]
