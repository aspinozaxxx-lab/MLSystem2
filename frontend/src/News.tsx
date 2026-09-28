import { ArrowLeft, ArrowUpRight } from "lucide-react";

type NewsImage = { src: string; alt: string; caption?: string };
type NewsArticle = {
  slug: string;
  date: string;
  title: string;
  summary: string;
  image?: NewsImage;
  sections: { title: string; paragraphs?: string[]; steps?: string[]; image?: NewsImage }[];
};

// Новые статьи добавляются первыми; адрес опубликованной статьи остаётся постоянным.
export const newsArticles: readonly NewsArticle[] = [{
  slug: "annotation-zones",
  date: "2026-09-28",
  title: "Размечайте часть снимка — обучайте только на ней",
  summary: "Теперь можно обвести территории, на которых разметка готова. Гровика использует для обучения только эти зоны, даже если исходный снимок гораздо больше.",
  image: {
    src: "/news/annotation-zones.svg",
    alt: "На большом снимке выделены две размеченные зоны. В тайле у границы зоны всё за её контуром остаётся пустым.",
    caption: "Схема: фиолетовый контур задаёт размеченную территорию, светлые клетки обозначают пустую часть тайла.",
  },
  sections: [
    {
      title: "Когда это удобно",
      paragraphs: ["Если на большом снимке размечен только один квартал, раньше неразмеченные участки могли попадать в обучение как фон. Теперь достаточно обозначить квартал размеченной зоной. На одном снимке можно нарисовать несколько зон.",
        "Внутри зоны разметка должна быть полной: все нужные объекты отмечены, остальная территория считается фоном. Сама зона не является объектом, который должна находить сеть."],
    },
    {
      title: "Как добавить зону",
      steps: [
        "Откройте «Редактор датасетов», выберите датасет и нужный снимок.",
        "Нажмите «Размеченная зона». Обведите готовую территорию кликами и завершите контур двойным щелчком.",
        "В объединённом датасете выберите нужный класс или «Все классы». Обычному датасету этот выбор не требуется.",
        "При необходимости добавьте другие зоны. Для исправления контура переключитесь на выбор и измените его вершины. Работают удаление и отмена, в том числе в полноэкранном режиме.",
        "Дождитесь сохранения черновика и нажмите «Опубликовать». После публикации запускайте обучение на обновлённой версии датасета.",
      ],
    },
    {
      title: "Что произойдёт при обучении",
      paragraphs: ["Каждая зона обрабатывается отдельно. Если тайл выходит за её контур, эта часть становится nodata: соседнее изображение туда не подставляется и в оценке качества не участвует. Исходный снимок при этом не меняется.",
        "Для объединённого датасета используется территория, размеченная для всех классов. Если у какого-то класса зон нет, для него размеченным считается весь снимок. Пересекающиеся зоны остаются отдельными: общая территория будет использована повторно.",
        "Зоны поддерживаются в legacy, next-gen2 и object f1. Они ограничивают обучение и его проверочные части, а обычное распознавание и создание псевдоразметки по-прежнему выполняются на полном снимке."],
    },
    {
      title: "Если зоны не нужны",
      paragraphs: ["Ничего добавлять не требуется: без зон используется весь снимок, как раньше. Удаление последней зоны возвращает этот режим — в редакторе появится надпись «Используется весь снимок».",
        "Кнопка «Зоны разметки» только показывает или скрывает контуры на карте. Она не отключает зоны для обучения. Старые обучения и их результаты остаются прежними."],
    },
  ],
}];

function NewsFigure({ image }: { image: NewsImage }) {
  return <figure className="news-figure">
    <img src={image.src} alt={image.alt} loading="lazy" />
    {image.caption ? <figcaption>{image.caption}</figcaption> : null}
  </figure>;
}

function articleDate(value: string): string {
  return new Date(`${value}T12:00:00`).toLocaleDateString("ru-RU", { day: "numeric", month: "long", year: "numeric" });
}

export function NewsSection() {
  return <section className="panel news-section" aria-labelledby="news-heading">
    <div className="panel-header"><div><h2 id="news-heading">Новости</h2><p>Что появилось в Гровике и как этим пользоваться</p></div></div>
    <div className="news-list">{newsArticles.map((article) => <a className="news-card" key={article.slug} href={`#/news/${article.slug}`}>
      <div className="news-card-copy">
        <time dateTime={article.date}>{articleDate(article.date)}</time>
        <h3>{article.title}</h3><p>{article.summary}</p>
        <span className="news-read">Как пользоваться <ArrowUpRight size={16} /></span>
      </div>
      {article.image ? <img src={article.image.src} alt="" loading="lazy" /> : null}
    </a>)}</div>
  </section>;
}

export function NewsPage({ slug }: { slug?: string }) {
  const article = newsArticles.find((item) => item.slug === slug);
  if (!slug) return <NewsSection />;
  if (!article) return <section className="panel"><h1>Новость не найдена</h1><a href="#/">На главную</a></section>;
  return <article className="panel news-article">
    <a className="news-back" href="#/"><ArrowLeft size={16} /> Все новости на главной</a>
    <header><time dateTime={article.date}>{articleDate(article.date)}</time><h1>{article.title}</h1><p className="news-lead">{article.summary}</p></header>
    {article.image ? <NewsFigure image={article.image} /> : null}
    {article.sections.map((section) => <section key={section.title}>
      <h2>{section.title}</h2>
      {section.paragraphs?.map((text) => <p key={text}>{text}</p>)}
      {section.steps ? <ol>{section.steps.map((text) => <li key={text}>{text}</li>)}</ol> : null}
      {section.image ? <NewsFigure image={section.image} /> : null}
    </section>)}
    <a className="primary" href="#/dataset-editor">Открыть редактор датасетов <ArrowUpRight size={16} /></a>
  </article>;
}
