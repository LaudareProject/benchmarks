/*
 * app.js - DOM wiring for the Laudare experiment matrix.
 * All generation logic lives in matrix.js; this file only renders and tracks state.
 */
(function () {
  'use strict';

  var M = LaudareMatrix;

  var selected = new Set();
  var cellInputs = new Map();      // cell key -> input element
  var masterInputs = new Map();    // column id -> input element

  var totalEnabled = 0;
  M.ROWS.forEach(function (row) {
    totalEnabled += M.enabledCellsInRow(row.id).length;
  });

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  function inputId(rowId, colId) {
    return 'cell--' + rowId + '--' + colId;
  }

  /* ------------------------------------------------------------- rendering */

  function buildHead() {
    var thead = document.getElementById('matrix-head');
    var groupRow = document.createElement('tr');
    groupRow.className = 'group-row';

    var corner = el('th', 'corner', 'Framework · Task');
    corner.rowSpan = 2;
    corner.scope = 'col';
    groupRow.appendChild(corner);

    M.COLUMN_GROUPS.forEach(function (group) {
      var th = el('th', 'group-cell', group.label);
      th.colSpan = group.columns.length;
      th.scope = 'colgroup';
      groupRow.appendChild(th);
    });
    thead.appendChild(groupRow);

    var columnRow = document.createElement('tr');
    columnRow.className = 'column-row';
    M.COLUMNS.forEach(function (column) {
      var th = document.createElement('th');
      th.className = 'column-cell';
      th.scope = 'col';

      var label = el('label', 'column-toggle');
      label.title = 'Toggle every enabled cell in column ' + column.letter + ' (' + column.label + ')';

      var box = document.createElement('input');
      box.type = 'checkbox';
      box.id = 'col--' + column.id;
      box.setAttribute('aria-label', 'Select all enabled cells in column ' + column.letter + ': ' + column.label);
      box.addEventListener('change', function () {
        toggleMany(M.enabledCellsInColumn(column.id), box.checked);
      });

      label.appendChild(box);
      label.appendChild(el('span', 'column-text', column.letter + ' · ' + column.label));
      th.appendChild(label);
      masterInputs.set(column.id, box);
      columnRow.appendChild(th);
    });
    thead.appendChild(columnRow);
  }

  function buildBody() {
    var tbody = document.getElementById('matrix-body');
    M.TASK_GROUPS.forEach(function (group) {
      var rows = M.ROWS.filter(function (row) { return row.task === group.task; });
      if (rows.length === 0) return;

      var groupRow = document.createElement('tr');
      groupRow.className = 'task-row';
      var groupCell = el('th', 'task-cell', group.label + ' (' + rows.length + ' frameworks)');
      groupCell.colSpan = M.COLUMNS.length + 1;
      groupCell.scope = 'colgroup';
      groupRow.appendChild(groupCell);
      tbody.appendChild(groupRow);

      rows.forEach(function (row) {
        var tr = document.createElement('tr');
        tr.className = 'model-row';
        tr.dataset.rowId = row.id;

        var labelCell = document.createElement('th');
        labelCell.className = 'row-label';
        labelCell.scope = 'row';

        var button = el('button', 'row-toggle', row.label);
        button.type = 'button';
        button.title = 'Toggle every enabled cell in this row';
        button.setAttribute('aria-label', 'Toggle every enabled cell for ' + row.label);
        button.addEventListener('click', function () {
          var keys = M.enabledCellsInRow(row.id);
          var allSelected = keys.length > 0 && keys.every(function (key) { return selected.has(key); });
          toggleMany(keys, !allSelected);
        });
        labelCell.appendChild(button);
        labelCell.appendChild(el('span', 'row-meta', row.framework + ' · ' + row.task + (row.model ? ' · ' + row.model : '')));
        tr.appendChild(labelCell);

        M.COLUMNS.forEach(function (column) {
          var key = M.cellKey(row.id, column.id);
          var reason = M.disabledReason(row, column);
          var td = document.createElement('td');
          td.className = 'cell' + (reason ? ' disabled' : '');
          if (reason) td.title = reason;

          var label = el('label', 'cell-toggle');
          label.title = reason || (row.label + ' — ' + column.label);

          var box = document.createElement('input');
          box.type = 'checkbox';
          box.id = inputId(row.id, column.id);
          box.setAttribute('aria-label', row.label + ', ' + column.label + (reason ? ' (not applicable)' : ''));
          box.dataset.rowId = row.id;
          box.dataset.colId = column.id;
          if (reason) {
            box.disabled = true;
            box.checked = false;
            box.title = reason;
          } else {
            cellInputs.set(key, box);
          }

          label.appendChild(box);
          td.appendChild(label);
          tr.appendChild(td);
        });

        tbody.appendChild(tr);
      });
    });
  }

  function buildLegend() {
    var list = document.getElementById('legend-list');
    M.LEGEND.forEach(function (text) {
      list.appendChild(el('li', null, text));
    });
  }

  /* ----------------------------------------------------------- interaction */

  function toggleMany(keys, checked) {
    keys.forEach(function (key) {
      if (checked) selected.add(key);
      else selected.delete(key);
    });
    syncUI();
  }

  function updateMasters() {
    M.COLUMNS.forEach(function (column) {
      var box = masterInputs.get(column.id);
      if (!box) return;
      var keys = M.enabledCellsInColumn(column.id);
      var count = 0;
      keys.forEach(function (key) { if (selected.has(key)) count += 1; });
      box.checked = keys.length > 0 && count === keys.length;
      box.indeterminate = count > 0 && count < keys.length;
      box.setAttribute('aria-checked', box.indeterminate ? 'mixed' : String(box.checked));
    });
  }

  function syncUI() {
    cellInputs.forEach(function (box, key) {
      box.checked = selected.has(key);
    });
    updateMasters();

    var count = selected.size;
    document.getElementById('selected-count').textContent =
      'Selected: ' + count + ' of ' + totalEnabled + ' cells';

    document.getElementById('generate').disabled = count === 0;

    var hint = document.getElementById('hint');
    if (count === 0) {
      hint.textContent = 'Select at least one cell to enable generation.';
    } else {
      hint.textContent = count + ' command(s) will be written to ' + M.SCRIPT_FILENAME + '.';
    }
  }

  function downloadScript() {
    if (selected.size === 0) return;
    var script = M.buildScript(selected);
    var blob = new Blob([script], { type: 'text/x-shellscript' });
    var url = URL.createObjectURL(blob);
    var anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = M.SCRIPT_FILENAME;
    anchor.rel = 'noopener';
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    window.setTimeout(function () { URL.revokeObjectURL(url); }, 5000);

    document.getElementById('hint').textContent =
      'Generated ' + M.SCRIPT_FILENAME + ' with ' + selected.size +
      ' command(s). Run chmod +x ' + M.SCRIPT_FILENAME + ' before executing it.';
  }

  function clearAll() {
    selected.clear();
    syncUI();
  }

  /* ----------------------------------------------------------------- boot */

  buildHead();
  buildBody();
  buildLegend();

  document.getElementById('matrix-body').addEventListener('change', function (event) {
    var box = event.target;
    if (!box || box.type !== 'checkbox' || !box.dataset.rowId) return;
    var key = M.cellKey(box.dataset.rowId, box.dataset.colId);
    if (box.checked) selected.add(key);
    else selected.delete(key);
    syncUI();
  });

  document.getElementById('generate').addEventListener('click', downloadScript);
  document.getElementById('clear').addEventListener('click', clearAll);

  syncUI();
})();
