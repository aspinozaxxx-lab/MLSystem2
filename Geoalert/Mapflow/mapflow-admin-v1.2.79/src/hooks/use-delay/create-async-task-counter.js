import { Observable, merge, Subject, timer, combineLatest } from "rxjs";
import {
  mapTo,
  scan,
  startWith,
  distinctUntilChanged,
  shareReplay,
  filter,
  pairwise,
  switchMap,
  takeUntil,
} from "rxjs/operators";

function toObservable(show, hide) {
  return new Observable(() => {
    show();
    return () => {
      hide();
    };
  });
}

export function createAsyncTaksCounter({
  show,
  hide,
  onTasksCountChange,
  flashTreshold = 250,
  minLoadingTime = 0,
}) {
  const taskStarts = new Subject();
  const taskCompletions = new Subject();

  const loadUp = taskStarts.pipe(mapTo(1));
  const loadDown = taskCompletions.pipe(mapTo(-1));
  const loadVariations = merge(loadUp, loadDown);
  const currentLoadCount = loadVariations.pipe(
    startWith(0),
    scan((total, change) => {
      const next = total + change;
      return next < 0 ? 0 : next;
    }),
    distinctUntilChanged(),
    shareReplay({ bufferSize: 1, refCount: true }),
  );
  const spinnerDeactivated = currentLoadCount.pipe(
    filter((count) => count === 0),
  );
  const spinnerActivated = currentLoadCount.pipe(
    pairwise(),
    filter(([prev, curr]) => prev === 0 && curr === 1),
  );

  const flashTresholdTimer = timer(flashTreshold);
  const minLoadingTimeTimer = timer(minLoadingTime);

  const shouldShowSpinner = spinnerActivated.pipe(
    switchMap(() => flashTresholdTimer.pipe(takeUntil(spinnerDeactivated))),
  );
  const shouldHideSpinner = combineLatest(
    minLoadingTimeTimer,
    spinnerDeactivated,
  );
  const showSpinner = toObservable(show, hide);
  const subscription = shouldShowSpinner
    .pipe(switchMap(() => showSpinner.pipe(takeUntil(shouldHideSpinner))))
    .subscribe();

  let counterSubscription;
  if (onTasksCountChange)
    counterSubscription = currentLoadCount.subscribe(onTasksCountChange);

  return {
    startTask() {
      taskStarts.next();
    },
    completeTask() {
      taskCompletions.next();
    },
    clear() {
      subscription.unsubscribe();
      if (counterSubscription) counterSubscription.unsubscribe();
    },
  };
}
