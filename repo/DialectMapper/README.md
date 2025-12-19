# Dialect mapper

The CSV on which this repo relies has been officially published throught the National Library of Norway's Language Bank. It can be accessed [here](https://www.nb.no/sprakbanken/en/resource-catalogue/oai-nb-no-sbr-92/).

If you find this code or dialect mappings useful, please cite [Adding Metadata to Existing Parliamentary Speech Corpus](https://aclanthology.org/2025.nodalida-1.49/)

## General information

Dialectal information given about speakers can vary between corpora. Some corpora may record the municipality of birth, or the county, or others may record the speakers' self-assessed dialect. We'd like a way of standardizing, or creating a meta-corpora mapping, between dialects and counties and municipalities. 

This package is bascially a wrapper around a file mapping between counties, municipalities, and dialects

## County/municipality to dialect mapping development

The alignment between kommune (municipalities), fylke (counties), and dialects was developed through manual effort. It began by looking over dialect maps (such as can be found in Skjekkeland's _Dei norske dialektane_ (page 276)) and attempting to align the currently existing kommune and fylke to them. Once I was satisified with the mapping I extended it to historical kommuner and fylker. To do this I referenced Wikipeida pages for the historial fylke and which kommune they contained and updated the list. The new 2024 updates were created via programtically parsing information in the table on the Regjering website [Nye kommune og fylkesnummer](https://www.regjeringen.no/no/tema/kommuner-og-regioner/kommunestruktur/nye-kommune-og-fylkesnummer-fra-1.-januar-2024/id2924701/). 

The named dialects used are from Skjekkeland's book (with the 4 Eastern regions collapsed into one). Additionally, the names were translated from Nynorsk to Bokmål. The numbered dialects are from a survey produced by Speech-Dat and used in the NST (? I only have the vaguest memory of this and I not can't find sources to back me up). 

The first column in the CSV file contains the historic kommuner and the second columns the modern kommuner. Where possible this creates a mapping between historic and current kommuner. That is, the kommune in column one became the kommune in column two. Occasionally, column one is left empty. This is for instances where a new kommune, one that does not have a historical equivalent, was included. The same aligment through time can be found for the three fylke columns as well. 


## Pip installation

This package can be pip installed and thus imported into code regarless of it's relationship to this directory. 

To pip install 

1. Nativate to this `dialect_mapper` directory
2. Run `pip install .` 

## Code usage

The mapper can be inported into other Python code and used like so

```python
import dialect_mapper

# you need to instaniate the class so it reads the CSV file
mm = dialect_mapper.mapper_methods()

mm.get_named_dialect('Arendal')
```

The `get_named_dialect()` and `get_numeric_dialect()` methods try to match the input against old municipalities, new municipalities, old counties, and new counties. If you know specifcially what input you're using, you can use a more explicit method such as `get_named_dialect_by_old_municipality()`

### Less fine-grained of dialects

While we have provided a relatively fine-grained mapping we may not always want/need such detail. Therefore there are two methods of collapsing regions into larger ones

1) The first method is to call the `enable_fine_grained_dialect_collapse`. This will collapse the different Mid dialects into just Trøndsk as well as transform Midlandsk into Østlandsk during the mapping process.

2) **NOTE** This method has been deprecated. Please see the cardinal dialect regions secion below. Old documentation: The second method is to use the `get_cardinal_dialect`. This is meant for use either post- or ad-hoc. That is, you should use the named mapping function(s) normally then pass the resulting names through this function. Additionally, this method can call the `get_named_dialect` method on its own so it is possible to pass a kommune or fylke name and get a cardinal dialect back. 

### Cardinal dialect regions

Support has been added for the cardinal (e.g. North, Mid, etc.) dialect regions. The `get_cardinal_five()` method(s) return one of the five cardinal dialect regions (that is, North, Mid, West, East, and South). The `get_cardinal_four()` method(s) work similarly only the South region has been removed. 

## Special mappings

As is inevitable when humans are inputting data, there are some typos or other inconsistencies in the location data for certain speakers in various corpora. I've done my best to manually correct these and make them available in this package. To enable the corrections simply call the enable method on the `mapper_methods` object before querrying for the named, numeric, or cardinal dialect. 

**NB Tale**
```python
import dialect_mapper

mm = dialect_mapper.mapper_methods()
mm.enable_nbtale_corrections()
named_dialect = mm.get_named_dialect('Hyllestand')
```

**NPSC**

```python
import dialect_mapper

mm = dialect_mapper.mapper_methods()
mm.enable_npsc_corrections()
named_dialect = mm.get_named_dialect('Vestfossen')
```

**Stortinget (SSC)**
```python
import dialect_mapper

mm = dialect_mapper.mapper_methods()
mm.enable_stortinget_corrections()
named_dialect = mm.get_named_dialect('opdal')
```

**Nordic Dialect Corpus (NDC)**
```python
import dialect_mapper

mm = dialect_mapper.mapper_methods()
mm.enable_ndc_corrections()
named_dialect = mm.get_named_dialect('Kirkenær')
```
