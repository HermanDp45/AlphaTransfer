import dataAppHtml from "../dist/index.html?raw";
import seedSnapshot from "./data.json";
import { dataAppOwnerEmailSha256 } from "./data-app-owner.js";
import { createDataAppWorker } from "./data-app-worker.js";

export default createDataAppWorker({
  html: dataAppHtml,
  seedSnapshot,
  ownerEmailSha256: dataAppOwnerEmailSha256,
});
