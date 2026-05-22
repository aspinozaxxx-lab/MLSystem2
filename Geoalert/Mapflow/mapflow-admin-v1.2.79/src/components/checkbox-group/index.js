import { Checkbox, FormGroup } from "@blueprintjs/core";
import React, { useEffect } from "react";
import { useWatch, useController } from "react-hook-form";

export const CheckboxGroup = ({
  control,
  label,
  labelInfo,
  name,
  options,
  ...rest
}) => {
  const {
    field: { ref, value, onChange, ...inputProps },
  } = useController({
    name,
    control,
    defaultValue: [],
  });

  const checkboxIds = useWatch({ control, name }) || [];

  const handleChange = (value) => {
    const newArray = [...checkboxIds];
    const item = value;

    const index = newArray.findIndex((x) => x === item);

    // If theres no match add item to the array
    if (index === -1) {
      newArray.push(item);
    } else {
      //If there is a match and the value is empty, remove the item from the array
      newArray.splice(index, 1);
    }

    onChange(newArray);
  };
  useEffect(() => {
    const initialCheckedItems = options
      .filter((option) => option.defaultEnabled)
      .map((option) => option.name);
    onChange(initialCheckedItems);
  }, []);

  return (
    <div>
      <FormGroup label={label} labelInfo={labelInfo}>
        {options.map((option) => (
          <Checkbox
            key={option.name}
            defaultChecked={option.defaultEnabled}
            checked={value?.includes(option.name)}
            {...inputProps}
            inputRef={ref}
            onChange={() => handleChange(option.name)}
            disabled={rest?.disabled}
            label={option.displayName || option.name}
          />
        ))}
      </FormGroup>
    </div>
  );
};
