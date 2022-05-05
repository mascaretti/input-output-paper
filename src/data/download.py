import pycurl
import os
import glob
from src.data.utils import get_eu_countries

def _download_national_iot(url, output_file):
    '''This function download a file from a given
    URL and save it in output_file.'''
    with open(output_file, 'wb') as f:
        c  = pycurl.Curl()
        c.setopt(c.URL, url)
        c.setopt(c.WRITEDATA, f)
        c.perform()
        c.close()

def _download_countries(country_list, output_directory, download_behaviour):
    '''This function download the data for countries
    in a provided country list of ISO3 codes if the download behaviour is set to
    "overwrite".
    It only downloads the missing files when behaviour is "default".
    '''

    try:
        assert download_behaviour in ['default', 'overwrite']
    except AssertionError:
        print("Invalid download value.")
        raise

    for country in country_list:
        country_url = 'http://www.wiod.org/protected3/data16/niot/{0}_niot_nov16.xlsx'.format(country)
        country_output_file = os.path.join(output_directory, '{0}.xlsx'.format(country))
        if download_behaviour == "overwrite":
            print("Downloading data for {0}...".format(country))
            _download_national_iot(country_url, country_output_file)
        else:
            if os.path.isfile(country_output_file) == True:
                print("Data for {0} already downloaded!".format(country))
            else:
                print("Downloading data for {0}...".format(country))
                _download_national_iot(country_url, country_output_file)
        assert os.path.isfile(country_output_file) == True

def download_data(eu, output_directory, country_list=None, download_behaviour="default"):
    ''' This function download the countries data of EU countries
    if eu is set equal to True or, otherwise, of a list of countries
    defined by their ISO3 country code.
    '''

    download_list = get_eu_countries() if eu else country_list
    download_list.sort()
    _download_countries(download_list, output_directory, download_behaviour=download_behaviour)

def _check_file(country_name, file_list):
    '''This function checks whether a country is
    contained within a list of file names. It does so by
    using a boolean variable and logical OR so that we obtain at least
    one match.'''
    try:
        check_bool = False
        for file_item in file_list:
            check_bool = check_bool or (country_name.lower() in file_item.lower())
        assert check_bool == True
    except AssertionError:
        print("Data for {0} were not found!".format(country_name))
        print("Change download option!")
        print(" ******************* \n")
        raise

def _check_files(country_list, directory):
    '''This function checks whether the provided country
    list is present in memory.'''

    file_list = glob.glob(os.path.join(directory, "*.xlsx"))
    for country in country_list:
        print("Checking {0}...".format(country))
        _check_file(country, file_list)

def check_data(eu, output_directory, country_list=None):
    '''This function checks data are downloaded'''
    country_list = get_eu_countries() if eu else country_list
    _check_files(country_list, output_directory)

def check_country_list(country_list):
    '''This function checks the ISO3 code
    of any provided country list'''
    available_data = ['AUS',
    'AUT', 'BEL', 'BGR', 'BRA',
    'CAN', 'CHE', 'CHN', 'CYP',
    'CZE', 'DEU', 'DNK', 'ESP', 
    'EST', 'FIN', 'FRA', 'GBR',
    'GRC', 'HRV', 'HUN', 'IDN',
    'IND', 'IRL', 'ITA', 'JPN',
    'KOR', 'LTU', 'LUX', 'LVA',
    'MEX', 'MLT', 'NLD', 'NOR',
    'POL', 'PRT', 'ROU', 'RUS',
    'SVK', 'SVN', 'SWE', 'TUR',
    'TWN', 'USA']
    for country in country_list:
        try:
            assert country in available_data
        except AssertionError:
            print("\n*** {0} is not a valid country ISO3 code. Check the settings. ***\n".format(country))
            raise
