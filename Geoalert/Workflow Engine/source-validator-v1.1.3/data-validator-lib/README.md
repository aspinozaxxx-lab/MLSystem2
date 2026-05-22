# data-validator-lib

implementation of data requirements validation for mapflow api WDs.

See description here:

https://geoalert.fibery.io/Documentation/Data-requirements-proposal-649

# Library structure

## Factory
The outer interface is `get_validator()` function, which returns the class for validation, 
based on the validator name passed as parameter. The returned validator is a callable 
that returns a result with a status and a dict with problems log.

## Validator
Every Validator defined in the `validators` package must be inherited from `Validator`
and inherits the same `__call__` interface, which includes sequential call of 
three checks, which should be implemented in the subclasses:
- check if the data type in the `request` is allowed for the `wd`
- validate the `request` itself (that the link is of allowed format)
- check all other source parameters which are listed in the wd

The `__call__` has two parameters:
 - wd - dict (workflow definition or its part), following the description
https://geoalert.fibery.io/Documentation/Data-requirements-proposal-649
 - request, which depends on source type (will be described later)

## To add a new validator for `source_type`
- Add a file in data_validator_lib/validators with appropriate name
- Implement SourceTypeValidator(Validator) class in that file
- Import it in `__init__` as `source_type` - this will automatically allow to create it from factory with parameter `source_type`
- Impelment tests

