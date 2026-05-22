import { clone } from "./common";

function createShortcut(label, dateRange) {
  return { dateRange, label };
}

export function createDefaultShortcuts() {
  const today = new Date();

  const makeDate = (action) => {
    const returnVal = clone(today);
    action(returnVal);
    returnVal.setDate(returnVal.getDate() + 1);
    return returnVal;
  };

  const oneWeekAgo = makeDate((d) => d.setDate(d.getDate() - 7));
  const oneMonthAgo = makeDate((d) => d.setMonth(d.getMonth() - 1));
  const threeMonthsAgo = makeDate((d) => d.setMonth(d.getMonth() - 3));
  const sixMonthsAgo = makeDate((d) => d.setMonth(d.getMonth() - 6));
  const oneYearAgo = makeDate((d) => d.setFullYear(d.getFullYear() - 1));

  return [
    // Nothing selected - All
    createShortcut("Всё время", [null, null]),
    createShortcut("Сегодня", [today, today]),
    createShortcut("1 неделю назад", [oneWeekAgo, today]),
    createShortcut("1 месяц назад", [oneMonthAgo, today]),
    createShortcut("3 месяца назад", [threeMonthsAgo, today]),
    createShortcut("6 месяца назад", [sixMonthsAgo, today]),
    createShortcut("1 год назад", [oneYearAgo, today]),
  ];
}

export const DEFAULT_SHORTCUTS = createDefaultShortcuts();

export const MONTHS = [
  "Январь",
  "Февраль",
  "Март",
  "Апрель",
  "Май",
  "Июнь",
  "Июль",
  "Август",
  "Сентябрь",
  "Октябрь",
  "Ноябрь",
  "Декабрь",
];

export const WEEKDAYS = ["По", "Вт", "Ср", "Че", "Пя", "Су", "Во"];
