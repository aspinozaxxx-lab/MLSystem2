import { MultiSelect2 } from "@blueprintjs/select";
import { MenuItem } from "@blueprintjs/core";
import "normalize.css";
import "@blueprintjs/core/lib/css/blueprint.css";
import "@blueprintjs/select/lib/css/blueprint-select.css";
import { Trans } from "@lingui/react";

import {
  ProgressStatuses,
  STATUS_FAILED,
  STATUS_PENDING,
  STATUS_SUCCESS,
  STATUS_UNPROCESSED,
  STATUS_CANCELLED,
  TRANSLATIONS,
} from "constants/common";

const elements = [
  STATUS_SUCCESS,
  STATUS_FAILED,
  STATUS_PENDING,
  STATUS_UNPROCESSED,
  STATUS_CANCELLED,
];

export default function MultipleSelect({ items, setItems }) {
  const handleSelect = (el) => {
    setItems(
      !items.includes(TRANSLATIONS[el.target.textContent])
        ? [...items, TRANSLATIONS[el.target.textContent]]
        : items.filter((item) => item !== TRANSLATIONS[el.target.textContent]),
    );
  };

  return (
    <div style={{ maxWidth: 400 }}>
      <MultiSelect2
        activeItem={items}
        items={elements}
        selectedItems={items}
        popoverProps={{ minimal: true }}
        itemRenderer={(val, itemProps) => {
          return (
            <MenuItem
              key={val}
              text={<Trans id={ProgressStatuses.T[val]} />}
              onClick={(elm) => handleSelect(elm)}
              active={itemProps.modifiers.active}
              selected={items.includes(val)}
            />
          );
        }}
        placeholder="Выберите статусы..."
        tagRenderer={(item) => <Trans id={ProgressStatuses.T[item]} />}
        onRemove={(item) => {
          setItems((items) => items.filter((elm) => elm !== item));
        }}
        onClear={() => setItems([])}
      />
    </div>
  );
}
