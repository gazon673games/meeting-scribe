'use strict';

const electron = require('electron');
const { exposeMeetingScribeApi } = require('./preload-api.cjs');

const meetingScribeApi = exposeMeetingScribeApi(electron);

module.exports = { meetingScribeApi };
