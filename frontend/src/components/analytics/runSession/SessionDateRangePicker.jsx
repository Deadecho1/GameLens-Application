import DatePicker from 'react-datepicker';
import { enUS } from 'date-fns/locale/en-US';
import 'react-datepicker/dist/react-datepicker.css';
import './runSessionDatePicker.css';

function parseYmd(yyyyMmDd) {
  if (!yyyyMmDd) return null;
  const [y, m, d] = yyyyMmDd.split('-').map(Number);
  if (!y || !m || !d) return null;
  return new Date(y, m - 1, d);
}

function toYmd(date) {
  if (!(date instanceof Date) || Number.isNaN(date.getTime())) return '';
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, '0');
  const day = String(date.getDate()).padStart(2, '0');
  return `${y}-${m}-${day}`;
}

function SessionDateField({ id, label, selected, minDate, maxDate, disabled, onChange }) {
  return (
    <div className="flex flex-col gap-1.5" lang="en-US">
      <label htmlFor={id} className="font-display text-xs font-bold uppercase tracking-wider text-slate-400">
        {label}
      </label>
      <div className="gl-session-date-field relative w-fit">
        <DatePicker
          id={id}
          selected={selected}
          onChange={(date) => {
            if (date) onChange(toYmd(date));
          }}
          minDate={minDate}
          maxDate={maxDate}
          disabled={disabled}
          locale={enUS}
          dateFormat="MMMM d, yyyy"
          calendarStartDay={0}
          showPopperArrow={false}
          popperPlacement="bottom-start"
          calendarClassName="gl-session-datepicker"
          popperClassName="gl-session-datepicker-popper"
          className="gl-session-date-input font-data"
          placeholderText="Select date"
          autoComplete="off"
          aria-label={label}
        />
      </div>
    </div>
  );
}

/**
 * English-only calendar popup date range (From / To) for session stats filtering.
 */
export default function SessionDateRangePicker({
  from,
  to,
  minDate,
  maxDate,
  onFromChange,
  onToChange,
  disabled = false,
}) {
  const datasetMin = parseYmd(minDate);
  const datasetMax = parseYmd(maxDate);
  const fromSelected = parseYmd(from);
  const toSelected = parseYmd(to);

  const fromMax = toSelected && datasetMax
    ? new Date(Math.min(toSelected.getTime(), datasetMax.getTime()))
    : datasetMax;
  const toMin = fromSelected && datasetMin
    ? new Date(Math.max(fromSelected.getTime(), datasetMin.getTime()))
    : datasetMin;

  return (
    <div className="flex flex-wrap items-end gap-4" lang="en-US">
      <SessionDateField
        id="run-session-date-from"
        label="From"
        selected={fromSelected}
        minDate={datasetMin}
        maxDate={fromMax}
        disabled={disabled || !datasetMin}
        onChange={onFromChange}
      />
      <SessionDateField
        id="run-session-date-to"
        label="To"
        selected={toSelected}
        minDate={toMin}
        maxDate={datasetMax}
        disabled={disabled || !datasetMax}
        onChange={onToChange}
      />
    </div>
  );
}
